#pragma once
#include <roaring/roaring.h>
#include <cstdint>
#include <random>

namespace cu_roaring::bench {

inline roaring_bitmap_t* generate_bitmap(uint32_t universe_size,
                                         double density,
                                         uint64_t seed = 42) {
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (uint32_t i = 0; i < universe_size; ++i) {
        if (dist(gen) < density) {
            roaring_bitmap_add(r, i);
        }
    }
    roaring_bitmap_run_optimize(r);
    return r;
}

inline roaring_bitmap_t* generate_clustered_bitmap(
    uint32_t universe_size,
    double   density,
    uint32_t n_clusters,
    double   cluster_density,
    double   background_density,
    uint64_t seed = 42)
{
    roaring_bitmap_t* r = roaring_bitmap_create();
    std::mt19937 gen(seed);

    // Determine target cardinality
    uint64_t target_card = static_cast<uint64_t>(universe_size * density);

    // Place clusters randomly
    std::uniform_int_distribution<uint32_t> pos_dist(0, universe_size - 1);
    uint32_t cluster_size = target_card > 0
        ? static_cast<uint32_t>(target_card / n_clusters / cluster_density)
        : 1000;

    struct Cluster {
        uint32_t start;
        uint32_t end;
    };
    std::vector<Cluster> clusters(n_clusters);
    for (uint32_t c = 0; c < n_clusters; ++c) {
        uint32_t s = pos_dist(gen);
        uint32_t e = std::min(s + cluster_size, universe_size);
        clusters[c] = {s, e};
    }

    // Generate bitmap
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    for (uint32_t i = 0; i < universe_size; ++i) {
        bool in_cluster = false;
        for (auto& cl : clusters) {
            if (i >= cl.start && i < cl.end) {
                in_cluster = true;
                break;
            }
        }
        double p = in_cluster ? cluster_density : background_density;
        if (prob(gen) < p) {
            roaring_bitmap_add(r, i);
        }
    }

    roaring_bitmap_run_optimize(r);
    return r;
}

}  // namespace cu_roaring::bench
