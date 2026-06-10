/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MRM construction property tests (§8 of the design doc):
 *  (a) decoded key/id set == union of inputs
 *  (b) per-id mask == per-lane membership vector
 *  (c) container types behave as designed (RUN_MASKED on constant-mask
 *      runs, BITMAP_MASKED on dense unions, ARRAY_MASKED otherwise)
 * covering every input container-type pairing (array/bitmap/run), empty
 * inputs, 65535/65536 boundaries, and shared-filter (m>1) workloads.
 */

#include <gtest/gtest.h>

#include <cu_roaring/cu_roaring.cuh>
#include <cu_roaring_mrm/mrm.cuh>

#include <roaring/roaring.h>

#include <cstdint>
#include <map>
#include <random>
#include <vector>

using cu_roaring::GpuRoaring;
using cu_roaring::mrm::mask_t;
using cu_roaring::mrm::Mrm;
using cu_roaring::mrm::MrmContainerType;

namespace {

// CPU reference: id -> k-bit membership mask.
std::map<uint32_t, mask_t> reference(const std::vector<std::vector<uint32_t>>& lanes)
{
  std::map<uint32_t, mask_t> ref;
  for (size_t l = 0; l < lanes.size(); ++l)
    for (uint32_t id : lanes[l])
      ref[id] |= mask_t{1} << l;
  return ref;
}

struct Fixture {
  std::vector<GpuRoaring> bitmaps;
  std::vector<roaring_bitmap_t*> croarings;

  ~Fixture()
  {
    for (auto& b : bitmaps)
      cu_roaring::gpu_roaring_free(b);
    for (auto* r : croarings)
      roaring_bitmap_free(r);
  }

  // Upload via the sorted-ids GPU path (produces ARRAY / BITMAP containers).
  void add_ids(const std::vector<uint32_t>& ids, uint32_t universe)
  {
    bitmaps.push_back(cu_roaring::upload_from_sorted_ids(
      ids.data(), static_cast<uint32_t>(ids.size()), universe));
  }

  // Upload via CRoaring with run_optimize (produces RUN containers for ranges).
  void add_croaring(const std::vector<uint32_t>& ids)
  {
    roaring_bitmap_t* r = roaring_bitmap_create();
    for (uint32_t id : ids)
      roaring_bitmap_add(r, id);
    roaring_bitmap_run_optimize(r);
    croarings.push_back(r);
    bitmaps.push_back(cu_roaring::upload(r));
  }
};

void check_against_reference(const Mrm& m, const std::vector<std::vector<uint32_t>>& lanes)
{
  auto ref = reference(lanes);
  std::vector<uint32_t> ids;
  std::vector<mask_t> masks;
  cu_roaring::mrm::mrm_decode(m, ids, masks);

  ASSERT_EQ(ids.size(), ref.size()) << "decoded id count != union size";
  size_t i = 0;
  for (auto& [id, mask] : ref) {
    ASSERT_EQ(ids[i], id) << "id mismatch at rank " << i;
    ASSERT_EQ(masks[i], mask) << "mask mismatch for id " << id;
    ++i;
  }
}

std::vector<uint32_t> random_ids(std::mt19937& rng, uint32_t universe, size_t n)
{
  std::vector<uint32_t> ids;
  std::map<uint32_t, bool> seen;
  while (ids.size() < n) {
    uint32_t v = rng() % universe;
    if (!seen.count(v)) {
      seen[v] = true;
      ids.push_back(v);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<uint32_t> range_ids(uint32_t start, uint32_t count)
{
  std::vector<uint32_t> ids(count);
  for (uint32_t i = 0; i < count; ++i)
    ids[i] = start + i;
  return ids;
}

uint8_t type_of_chunk(const Mrm& m, size_t chunk)
{
  std::vector<cu_roaring::mrm::MrmContainerDesc> descs(m.view.n_chunks);
  cudaMemcpy(descs.data(),
             m.view.descs,
             m.view.n_chunks * sizeof(cu_roaring::mrm::MrmContainerDesc),
             cudaMemcpyDeviceToHost);
  return descs[chunk].type;
}

}  // namespace

TEST(MrmBuild, SingleArrayLane)
{
  std::mt19937 rng(1);
  std::vector<std::vector<uint32_t>> lanes = {random_ids(rng, 1 << 20, 500)};
  Fixture f;
  f.add_ids(lanes[0], 1 << 20);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 1);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, ArrayArrayOverlap)
{
  std::mt19937 rng(2);
  std::vector<std::vector<uint32_t>> lanes = {random_ids(rng, 1 << 18, 800),
                                              random_ids(rng, 1 << 18, 800)};
  // force overlap: copy half of lane 0 into lane 1
  for (size_t i = 0; i < 400; ++i)
    lanes[1].push_back(lanes[0][i * 2]);
  std::sort(lanes[1].begin(), lanes[1].end());
  lanes[1].erase(std::unique(lanes[1].begin(), lanes[1].end()), lanes[1].end());

  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 1 << 18);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 2);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, ArrayBitmapMix)
{
  std::mt19937 rng(3);
  // lane 0 dense in chunk 0 (bitmap container), lane 1 sparse across chunks
  std::vector<std::vector<uint32_t>> lanes = {random_ids(rng, 65536, 8000),
                                              random_ids(rng, 1 << 19, 1000)};
  Fixture f;
  f.add_ids(lanes[0], 1 << 19);
  f.add_ids(lanes[1], 1 << 19);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 2);
  check_against_reference(m, lanes);
  // chunk 0 union >= 8000 -> BITMAP_MASKED
  EXPECT_EQ(type_of_chunk(m, 0), static_cast<uint8_t>(MrmContainerType::BITMAP_MASKED));
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, BitmapBitmapDense)
{
  std::mt19937 rng(4);
  std::vector<std::vector<uint32_t>> lanes = {random_ids(rng, 1 << 17, 30000),
                                              random_ids(rng, 1 << 17, 30000)};
  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 1 << 17);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 2);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, RunSourcesViaCRoaring)
{
  // contiguous ranges -> CRoaring run containers after run_optimize
  std::vector<std::vector<uint32_t>> lanes = {range_ids(100, 5000), range_ids(3000, 4000)};
  Fixture f;
  for (auto& l : lanes)
    f.add_croaring(l);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 2);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, SharedFilterBecomesRunMasked)
{
  // The multitenant payoff case: identical contiguous filters in all lanes
  // must collapse to constant-mask runs.
  auto base = range_ids(1000, 3000);
  std::vector<std::vector<uint32_t>> lanes(8, base);
  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 1 << 16);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 8);
  check_against_reference(m, lanes);
  EXPECT_EQ(type_of_chunk(m, 0), static_cast<uint8_t>(MrmContainerType::RUN_MASKED));
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, PerturbedSharedFilters)
{
  std::mt19937 rng(7);
  auto base = random_ids(rng, 1 << 18, 3000);
  std::vector<std::vector<uint32_t>> lanes;
  for (int l = 0; l < 16; ++l) {
    auto ids = base;
    // flip ~1%: drop some, add some
    for (size_t i = l; i < ids.size(); i += 97)
      ids.erase(ids.begin() + static_cast<long>(i % ids.size()));
    for (int a = 0; a < 30; ++a)
      ids.push_back(rng() % (1 << 18));
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    lanes.push_back(std::move(ids));
  }
  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 1 << 18);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 16);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, BoundaryIds)
{
  std::vector<std::vector<uint32_t>> lanes = {
    {0, 63, 64, 65535},          // chunk 0 edges
    {65536, 65537, 131071},      // chunk 1 edges
    {0, 65535, 65536, 131071}};  // straddling
  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 131072);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 3);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, EmptyLane)
{
  std::mt19937 rng(9);
  std::vector<std::vector<uint32_t>> lanes = {random_ids(rng, 1 << 16, 100), {}};
  Fixture f;
  f.add_ids(lanes[0], 1 << 16);
  f.add_ids(lanes[1], 1 << 16);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 2);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, FullChunk)
{
  // a completely full chunk 0; universe kept larger so the upload path's
  // >50%-density complement optimization (negated storage) doesn't fire
  std::vector<std::vector<uint32_t>> lanes = {range_ids(0, 65536)};
  Fixture f;
  f.add_ids(lanes[0], 1 << 18);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 1);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, SixtyFourLanes)
{
  std::mt19937 rng(11);
  std::vector<std::vector<uint32_t>> lanes;
  for (int l = 0; l < 64; ++l)
    lanes.push_back(random_ids(rng, 1 << 18, 200 + static_cast<size_t>(l) * 10));
  Fixture f;
  for (auto& l : lanes)
    f.add_ids(l, 1 << 18);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 64);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}

TEST(MrmBuild, MixedTypesAcrossLanes)
{
  std::mt19937 rng(13);
  std::vector<std::vector<uint32_t>> lanes = {
    random_ids(rng, 65536, 10000),   // bitmap container
    random_ids(rng, 1 << 18, 700),   // array containers
    range_ids(2000, 6000),           // run (via croaring)
  };
  Fixture f;
  f.add_ids(lanes[0], 1 << 18);
  f.add_ids(lanes[1], 1 << 18);
  f.add_croaring(lanes[2]);
  Mrm m = cu_roaring::mrm::mrm_build(f.bitmaps.data(), 3);
  check_against_reference(m, lanes);
  cu_roaring::mrm::mrm_free(m);
}
