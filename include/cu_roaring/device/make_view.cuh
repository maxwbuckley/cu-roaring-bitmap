#pragma once
#include "cu_roaring/device/roaring_view.cuh"
#include "cu_roaring/types.cuh"

namespace cu_roaring {

// Create a lightweight device view from a GpuRoaring.
// The view references the same device memory — do not free GpuRoaring while view is alive.
inline GpuRoaringView make_view(const GpuRoaring& bitmap)
{
  GpuRoaringView v{};
  v.keys           = bitmap.keys;
  v.types          = reinterpret_cast<const ContainerTypeD*>(bitmap.types);
  v.offsets        = bitmap.offsets;
  v.cardinalities  = bitmap.cardinalities;
  v.n_containers   = bitmap.n_containers;
  v.bitmap_data    = bitmap.bitmap_data;
  v.array_data     = bitmap.array_data;
  v.run_data       = bitmap.run_data;
  return v;
}

}  // namespace cu_roaring
