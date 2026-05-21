# GPU Parallel Shortest Path Algorithms — Literature Survey

Condensed from 5 papers. Organized by algorithm family, then by practical insights.

---

## 1. The Fundamental Tradeoff: Work Efficiency vs. Parallelism

All GPU SSSP algorithms sit on a spectrum between two extremes:

| Algorithm | Work Complexity | Parallelism | GPU Fit |
|-----------|----------------|-------------|---------|
| Dijkstra | O(v log v + e) | **None** (1 vertex/iteration) | Terrible |
| Bellman-Ford | O(ve) | **Maximum** (all vertices/iteration) | Good parallelism, awful work |
| Delta-stepping | O(v log v + e) avg | **Tunable** via delta | Sweet spot |

**Key insight** (Davidson 2014): For high-diameter graphs, Dijkstra is 1000x more work-efficient than Bellman-Ford. For low-diameter graphs (scale-free), the gap shrinks to ~2x. The optimal GPU algorithm must balance these.

**Bellman-Ford does 100-1000x more work than Dijkstra** but processes all vertices in parallel. On GPUs, the excess work often doesn't matter if the hardware is fully utilized. But on high-diameter graphs (road networks), the excess work dominates.

---

## 2. Algorithm Family: Workfront Sweep (Davidson 2014)

**Paper:** Davidson, Baxter, Garland, Owens. "Work-Efficient Parallel GPU Methods for Single-Source Shortest Paths." IPDPS 2014.

### Core Idea
Improve Bellman-Ford by maintaining a **workfront** (active vertex queue). Only process edges from vertices whose distances changed in the previous iteration.

### Data Structures
- **CSR format** for graph storage (row offsets + column values + weights). Total size: 2e + v.
- **Workfront queue**: vertex IDs of recently updated vertices
- **Distance array**: one float/int per vertex
- **Duplicate removal**: after each iteration, each vertex writes its workfront index to a lookup table keyed by vertex ID. Read back; if index matches, vertex is valid (owner). Otherwise marked invalid and compacted away.

### GPU Traversal Strategies (Critical Section)

Three strategies for processing adjacency lists of workfront vertices:

#### a) Cooperative Blocks (Group Blocking)
- A thread block cooperatively processes a set of vertices
- Edge list offsets loaded into shared memory
- Threads strip edges off the shared edge list
- Binary search to find originating vertex
- **Good for:** uniform degree graphs
- **Bad for:** high-degree vertices cause block-level load imbalance

#### b) CTA+Warp+Scan
Separates adjacency lists into three size categories:
- **> CTA size**: entire block cooperates on one vertex
- **> warp but < CTA**: warp cooperates
- **< warp**: threads iteratively load vertex ID + edge offset to shared memory pool

Benefit: specialized throughput per adjacency list size. Cost: three separate stages within one kernel, lose parallelism between stages.

#### c) Load-Balanced Partitioning (BEST OVERALL)
- Instead of grouping equal numbers of *vertices*, group equal numbers of *edges*
- Use efficient sorted search (binary search on scanned edge offsets) to find intersection of each block's edge-list range with the work queue
- Ensures perfect load balance between blocks AND within blocks
- **Winner across most graph types** — handles pathological cases (one vertex connected to 3M others)

### Work Organization: Three Variants

#### Workfront Sweep
- Simplest: just prune vertices whose distances didn't change
- Saves ~9.6x work vs Bellman-Ford
- Average parallelism: ~88K vertices/iteration
- Only 1.05x more iterations than B-F

#### Near-Far (most important for GPU SSSP)
- Split workfront into **Near Set** (dist < i*delta) and **Far Pile** (dist >= i*delta)
- Process only Near Set each iteration
- When Near Set empty: compact Far Pile, remove invalidated entries, split again with (i+1)*delta
- **Delta heuristic**: delta = c * w / d, where c = warp width (32), w = avg edge weight, d = avg degree
- Saves ~260x work vs B-F, parallelism ~17K, 1.66x more iterations
- **Best overall performer** in Davidson's evaluation
- Far Pile acts as lazy deletion — most entries become invalid as closer vertices are processed

#### Bucketing
- Partition into A buckets (A=10 works best)
- Use Thrust radix sort to distribute into buckets by distance
- Process smallest bucket; split output into bucketable + far pile
- Saves ~353x work vs B-F, but only ~8K parallelism, 2.27x more iterations
- **Worst performer** due to high reorganization overhead (82% of runtime is sorting)
- Missing GPU primitive: **multisplit** (partition by key) — would make bucketing competitive

### Performance Results (GTX 680)
- **Workfront Sweep and Near-Far outperform Bellman-Ford** by 1.3-14x on non-road graphs, and Dijkstra by 0.24-29.4x
- **Road networks**: GPU methods CANNOT outperform serial Dijkstra due to low parallelism (avg 100-200 vertices/iteration). This is expected and inherent.
- Near-Far on 17M-vertex rmat: 350 MTEPS (2.4-2.7x over CPU baselines)
- **Recommended hybrid**: Start with Workfront Sweep (parallelism low at start), switch to Near-Far when workfront large enough, switch back to Workfront Sweep at end.

### Key Lessons for Raster Graphs
- Raster graphs are **regular meshes** (similar to delaunay/msdoor datasets). Near-Far works well on these (1.74x over Workfront Sweep on msdoor).
- Load-balanced partitioning is less critical for raster graphs (degree is constant: 4-8 neighbors), but still preferred for robustness.

---

## 3. Algorithm Family: ADDS — Asynchronous Dynamic Delta-Stepping (Wang 2021)

**Paper:** Wang, Fussell, Lin. "A Fast Work-Efficient SSSP Algorithm for GPUs." PPoPP 2021.

**Current state of the art.** Average 2.9x speedup over Near-Far across 226 graphs. 3.5x on RTX 3090.

### Three Key Innovations over Near-Far

#### Innovation 1: Multiple Buckets with Dynamic Memory
- Near-Far uses only 2 buckets — extremely coarse priority queue
- ADDS uses **32 buckets** in a circular work queue
- Each bucket is a circular FIFO array with dynamically allocated blocks (64K 32-bit words each)
- Memory allocated from a **pool of pre-allocated GPU memory**; FIFO ordering makes alloc/dealloc simple
- Pointer array: high 16 bits = block index, low 16 bits = offset within block
- Direct-mapped translation caches in scratchpad for each WTB and MTB

#### Innovation 2: Asynchronous Delegation via Manager Thread Block (MTB)

The fundamental synchronization problem: multiple threads reading from and writing to the same buckets = MRMW (Multiple Reader Multiple Writer), which is not scalable on GPUs.

**Solution: Decoupled architecture**
- **Worker Thread Blocks (WTBs)**: process vertices, write new work items to buckets
- **Manager Thread Block (MTB)**: single block that reads buckets and assigns work to WTBs
- WTBs only WRITE to buckets (via atomic `resv_ptr` increment)
- MTB only READS from buckets (via `read_ptr`)
- Converts MRMW → SRMW (Single Reader Multiple Writer), much simpler

**SRMW Queue Protocol:**
- Each bucket has `resv_ptr` (atomically incremented by WTBs to claim write slots) and `read_ptr` (advanced by MTB)
- N-word **segments** with a **Write Completed Counter (WCC)**: WTB writes item, issues memory fence, atomically increments WCC. MTB checks WCC = N before reading segment.
- **Assignment Flags (AF)**: MTB sets AF for each idle WTB with work location + size. WTB polls its AF.
- **Completed Work Counter (CWC)**: WTB increments CWC per block when done. MTB checks CWC = resv_ptr to know bucket is empty.

**Termination**: MTB detects two consecutive sweeps of the work queue with no assigned work.

#### Innovation 3: Dynamic Delta Selection

Static delta (Near-Far: delta = c * W/D) is far from optimal for all graphs.

**ADDS dynamic mechanism:**
- MTB monitors **hardware utilization** (= number of work items assigned at any time)
- Sets upper/lower bounds on utilization based on total hardware threads and avg degree
- If utilization low → increase delta (more parallelism, less ordering)
- If utilization high → decrease delta (better ordering, less wasted work)
- **Lower bound on delta**: if tail bucket has >= 65% of total items, **clipping** occurs (vertices crammed into last bucket, losing all ordering). Delta must stay above this.
- Wait for utilization to settle after each delta change (settling time scales with delta)
- **Higher-frequency fine-grained adjustment**: vary number of high-priority buckets from which MTB assigns work

### Performance Results
- **226 graphs** from Lonestar 4.0 + SuiteSparse (>100K vertices, >1M edges)
- RTX 2080 Ti: **avg 2.9x** over Near-Far, avg 14.2x over CPU delta-stepping, avg 34.4x over serial Dijkstra
- RTX 3090: **avg 3.5x** over Near-Far
- Speedup largely **independent of graph degree or diameter** (dynamic delta adapts)
- Ablation: dynamic delta accounts for ~1.5x; async SRMW accounts for ~2.2x over 2-bucket static-delta

### Work Efficiency Analysis
- For 20% of graphs: ADDS does <0.75x the work of NF (significant savings via better ordering)
- For 44% of graphs: ADDS does 0.75-1.5x the work of NF (similar)
- For 36% of graphs: ADDS does >1.5x the work of NF (accepts more work for higher GPU utilization)

### Key Lessons for Raster Graphs
- Raster graphs have uniform degree (~8 for 8-connected grid) and moderate diameter. ADDS's dynamic delta would adapt well.
- The MTB delegation pattern is similar to what pyorps V4 already does with a persistent cooperative kernel + control buffers. The SRMW queue protocol could be adopted.
- For raster SSSP where the graph is implicit (no CSR), the bucket data structure still applies — but vertices are pixel indices.

---

## 4. Algorithm Family: Parallel Bucket Heap + Parallel Dijkstra (Berney 2023)

**Paper:** Berney, Iacono, Karsin, Sitchinava. "A Parallel Priority Queue with Fast Updates for GPU Architectures." arXiv:1908.09378v2, 2023.

### Core Contribution
A **cache-efficient parallel priority queue** (parBucketHeap) that provides O(1 + log d) amortized depth for ExtractMin, Update, Delete, and BulkUpdate operations, where d = max update batch size.

### Why This Matters
Delta-stepping avoids priority queues entirely — but it degrades on **high-diameter, dense graphs** because the number of buckets (and thus phases) grows with diameter. Berney shows that a true parallel Dijkstra with an efficient priority queue can beat delta-stepping on such graphs.

### The Parallel Bucket Heap Data Structure

**Hierarchical structure** with q = ceil(log_4(n/d)) + 1 levels, each having a **bucket** B_i and **signal buffer** S_i.

**Key invariant**: If B_0 is non-empty, it contains the minimum priority element. ExtractMin simply removes from B_0.

**Operations:**
- **BulkUpdate(U)**: up to d elements inserted into S_0 (sorted by value). O(1 + log d) depth.
- **ExtractMin**: remove from B_0. O(1 + log d) depth.
- **Resolve(i)**: combined Empty/Fill operation. Empties S_i into B_i (merge, dedup, split by priority). If B_i too small, fills from B_{i+1}. Non-adjacent levels can Resolve in parallel.

**Capacity**: |B_i| = d * 2^(2i+1), |S_i| = d * 2^(2i). Scaling by d ensures BulkUpdate of d elements fits in S_0.

### Parallel Execution Schedule
- **DAG of dependencies**: Resolve(i) depends on Resolve(i-1) every 4th call
- Non-adjacent levels can execute concurrently
- At most log_4(n/d) levels active at a time

### GPU Implementation Details
- Each thread block maps to one level of the parBucketHeap
- Shared memory = internal memory (PEM model: B = tw, where t = warps/block, w = warp size)
- **Inter-block synchronization**: each block has a designated global memory location. Blocks busy-wait until signaled. **Must launch fewer blocks than SMs** to avoid deadlock.
- Subroutines: Thrust Merge, CUB PrefixSums, CUB RadixSort (for Select)
- CUDA dynamic parallelism to launch subroutines from within thread blocks

### parDijkstra Algorithm
n rounds: (1) ExtractMin of minimum vertex u, (2) relax all outgoing edges (u,v) in parallel, (3) BulkUpdate with improved distances (batch size up to d = max degree).

### Performance Results (RTX 2080 Ti, Quadro M4000)
- **Target**: dense graphs with large diameter (n=30K vertices, diameter n-1)
- RTX 2080 Ti: **2.8x over Gunrock, 12x over ADDS** at 300M+ edges
- Quadro M4000: **5.4x over Gunrock** at 200M edges
- **Crossover point**: parDijkstra faster than Gunrock once edges > 80M (2080 Ti) or 40M (M4000)
- ADDS performance degrades badly on high-diameter graphs (too many phases)

### Key Lessons for Raster Graphs
- Raster graphs have **moderate** diameter (sqrt(N) for NxN grid), not the extreme n-1 diameter where parDijkstra excels.
- For typical pyorps rasters (1000x1000 to 5000x5000), diameter is 1000-5000 — ADDS/Near-Far should still be competitive.
- The BulkUpdate primitive is interesting: in raster SSSP, when processing a cell, you update 4-8 neighbors simultaneously. This maps directly to BulkUpdate with d=8.
- **Not practical for pyorps**: the data structure is complex, and the advantage only manifests on very dense, high-diameter graphs. Raster grids are sparse (degree 4-8).

---

## 5. Algorithm Family: Direct Delta-Stepping on GPU (Lesnikov 2016)

**Paper:** Lesnikov, Chernoskutov. "Performance analysis of Delta-stepping algorithm on CPU and GPU." CEUR-WS Vol-1662, 2016.

### Implementation
Straightforward GPU delta-stepping with one CUDA thread per vertex.

**4 kernels per iteration:**
1. Check if vertex belongs to current bucket → move to set R, create light edge relaxation requests
2. Relax light edges (apply relaxation requests)
3. Create heavy edge relaxation requests from set R
4. Relax heavy edges

**Bucket data structure**: Two arrays — one stores bucket index per vertex (-1 if not in any bucket), second stores bucket sizes.

**Graph storage**: CSR (compressed sparse row).

### Delta Selection Findings

**CPU (OpenMP, R-MAT graphs with avg degree 32):**
- Optimal delta ≈ 0.04-0.05 (close to theoretical 1/d = 1/32)
- CPU performance is **very sensitive** to delta — small changes cause large performance swings
- Delta-stepping outperforms both sequential Dijkstra and parallel Bellman-Ford

**GPU (Tesla K20Xm, same graphs):**
- Optimal delta ≈ **0.5** (10x larger than CPU optimal!)
- GPU performance is **much less sensitive** to delta — stable for delta >= 0.5
- With low delta: buckets too fine-grained → not enough vertices per bucket → low occupancy → bad GPU utilization
- With high delta (>= 1): all edges become "light" → algorithm degenerates to Bellman-Ford → maximum parallelism but poor work efficiency
- **Sweet spot**: delta = 0.5 gives good parallelism while maintaining some ordering

### Key Insight: Delta for GPU ≈ 10x Delta for CPU
This is because GPUs need **much larger frontiers** to achieve good occupancy. A larger delta puts more vertices in each bucket, giving more parallel work per phase.

For R-MAT scale 22 (4M vertices, 128M edges):
- GPU achieves ~8000 MTEPS at optimal delta
- CPU achieves ~100 MTEPS at optimal delta

### Key Lessons for Raster Graphs
- For raster graphs with edge weights in [1, max_cost]: delta should be chosen so each bucket contains enough cells to saturate the GPU
- If max_cost = 65535 (uint16), even delta = 100 gives 655 buckets — too many for a 500x500 raster
- Pyorps V4 uses delta = 2 * max_raster * max_cf * n_dirs — this is intentionally large, making most edges "light" and producing few buckets. Consistent with Lesnikov's finding.

---

## 6. Algorithm Family: Locality-Based Relaxation (Safari 2017)

**Paper:** Safari, Ebnenasir. "Locality-Based Relaxation: An Efficient Method for GPU-Based Computation of Shortest Paths." TTCS 2017.

### Core Idea
Instead of relaxing just immediate neighbors, each thread propagates relaxation **k hops deep** via DFS from its assigned vertex. This:
1. Reduces number of kernel launches (each launch propagates k levels instead of 1)
2. Reduces CPU-GPU communication (N launches without checking convergence flag)
3. Improves data locality (thread accesses nearby vertices in CSR, which are often contiguous in memory)
4. Reduces thread divergence (each thread has more work → fewer idle threads)

### Algorithm Details

**Thread-vertex affinity**: Each thread t handles **two** vertices: startV[2t] and startV[2t+1]. This:
- Halves thread count (reducing divergence)
- Improves memory coalescing (adjacent CSR entries loaded by same thread)
- Assigning >2 vertices shows no further improvement

**Two flag arrays** (double-buffered): FlagArray[0][v] and FlagArray[1][v]. In each kernel launch, threads read from FlagArray[i][v] and write to FlagArray[i⊕1][v]. Avoids read-write race conditions without synchronization.

**Host algorithm (Algorithm 6):**
1. Launch Kernel_1 for N iterations (no CPU-GPU communication during these)
2. Then repeat-until loop: launch Kernel_2 (same as Kernel_1 but communicates Flag back to CPU) until Flag = false
3. N chosen experimentally: N = ceil(typical_iterations / k)

**Kernel (Algorithm 7 - RelaxLocalityAndSetFrontier):**
- If FlagArray[i][u] = true: launch iterative DFS from u up to depth k
- At each visited vertex v via parent w: if (w,v) already relaxed, backtrack; else Relax(w,v)
- If v is at depth k: set FlagArray[i⊕1][v] = true (mark as new frontier)

**Only 1 atomic operation per kernel** (atomicMin for distance update).

### Performance Results (GeForce GT 630, 96 cores — WEAK GPU!)

Road networks up to 6.2M vertices, 15.3M arcs:
- **3.36x-5.77x speedup** over Harish et al.'s classic GPU BFS-style algorithm
- CalNev (1.9M vertices): 3.5 seconds vs Davidson's ~4s (GTX 680!), Boost ~588ms, LoneStar ~3.9s
- Western USA (6.2M vertices, 15.3M arcs): 4.9 seconds vs Harish's 24.7 seconds

**Optimal k = 4** for tested graphs (diminishing returns at k=5 due to thread saturation).

### Comparison Table
| Method | Space | Kernels/iter | CPU-GPU comm/iter | Atomics | Speedup over Harish |
|--------|-------|-------------|-------------------|---------|-------------------|
| Harish | 4V+2A | 2 | >=1 | 1 | baseline |
| Singh | 3V+2A | 1 | >=1 | 1 | 2.5x |
| Busato | 4V+2A | 2 | >=1 | 2 | 1.9-2.6x |
| **This work** | **4V+2A** | **1** | **<1** | **1** | **3.36-5.77x** |

### Key Lessons for Raster Graphs
- **Directly applicable to raster SSSP!** In a raster grid, k-hop DFS from a cell visits a diamond-shaped neighborhood. This is essentially what pyorps's Cython traversal does (process neighbors, which process their neighbors).
- The "N kernel launches without CPU communication" technique is similar to pyorps V4's persistent kernel — both avoid CPU-GPU round trips.
- The double-buffered flag array is a clean alternative to atomic flag operations.
- **For raster grids**: k-hop DFS naturally follows the grid structure. With 8-connected grids, k=3 means each thread explores up to ~8^3 = 512 cells (in theory), but DFS pruning via "already relaxed" check keeps it manageable.
- **Limitation**: This approach is best for **high-diameter, low-degree** graphs (road networks). On low-diameter, high-degree graphs, it provides less benefit.

---

## 7. Cross-Cutting Technical Insights

### 7.1 Atomic Operations
- **atomicMin for floats**: not supported in hardware on any current NVIDIA GPU. Must use software CAS loop (Gunrock 1.0 provides one). ADDS uses this.
- **atomicMin for integers**: hardware-supported since Kepler. Much faster.
- `__ballot_sync(0xFFFFFFFF)` fails silently in divergent code — use `atomicAdd` or `__activemask()` instead (confirmed by pyorps experience).

### 7.2 Memory Coalescing
All papers agree: **coalesced global memory access is the single most important GPU optimization for graph algorithms.**
- CSR column/weight arrays are accessed in scattered patterns — fundamentally bad for coalescing
- Raster grids have an advantage: neighbor indices are predictable (cell ± 1, cell ± width) — this enables better coalescing than general sparse graphs
- Shared memory should be used for frequently accessed data within a thread block

### 7.3 Synchronization
- **Intra-block**: `__syncthreads()` — fast, hardware-supported
- **Inter-block**: No hardware support (pre-cooperative groups). Software barriers via global memory atomic counters + busy-wait. **Deadlock risk**: must launch <= SM count blocks.
- **grid.sync()** (cooperative groups): available since Volta. pyorps experience: does NOT provide proper memory ordering on Blackwell/sm_120. Use custom atomic barrier with `__threadfence()`.
- **BSP model** (bulk synchronous parallel): each iteration = one kernel launch. Simple but forces CPU-GPU sync per iteration. Near-Far uses this.
- **Asynchronous** (ADDS, pyorps V4): persistent kernel avoids kernel launch overhead. Requires careful synchronization.

### 7.4 Delta Selection Summary

| Paper | Recommended Delta | Context |
|-------|------------------|---------|
| Meyer & Sanders (1998) | delta = Theta(1/d) | Theory, arbitrary graphs |
| Lesnikov (2016) CPU | delta ≈ 1/d | R-MAT, avg degree 32 |
| Lesnikov (2016) GPU | delta ≈ 0.5 (~16/d) | R-MAT, avg degree 32, K20Xm |
| Davidson (2014) | delta = c*w/d (c=32) | Near-Far heuristic |
| Wang (2021) ADDS | **Dynamic** (runtime) | Adapts per-graph |
| pyorps V4 | 2 * max_raster * max_cf * n_dirs | Raster grids, intentionally large |

**Universal finding**: GPU needs 10-30x larger delta than CPU to achieve good occupancy. Static heuristics are suboptimal — dynamic (ADDS) is best when feasible.

### 7.5 Graph Storage
- **General graphs**: CSR (Compressed Sparse Row) is universal. 3 arrays: row_offsets[v+1], col_indices[e], weights[e].
- **Raster grids**: No explicit graph needed! Neighbors computed from cell index + grid dimensions. This saves massive memory (no CSR arrays) and enables better cache behavior.

### 7.6 Work Organization Taxonomy

```
                    More Work-Efficient ←————→ More Parallel

Dijkstra ─── Delta-step ─── Near-Far ─── Workfront Sweep ─── Bellman-Ford
  (1 vtx)    (1 bucket)    (2 buckets)   (all updated)        (all vtx)
                 ↑
              ADDS (32 buckets, dynamic delta)
              parDijkstra (full priority queue)
```

### 7.7 Frontier Management
| Method | Frontier | Pros | Cons |
|--------|----------|------|------|
| Flag array (1 bit/vertex) | O(V) | No duplicates, O(1) check | Must scan all V bits to find frontier |
| Explicit queue | O(frontier) | Only process active vertices | Duplicates possible, atomic append |
| Double-buffered queue | O(frontier) | Avoids read-write conflicts | 2x memory, forces BSP |
| Bucket queue (ADDS) | O(active) | Priority ordering | Complex memory management |

For **raster grids**: flag array is attractive because scanning V cells is fast (contiguous memory, vectorizable). This is essentially what pyorps V4 does with its pending/distance arrays.

---

## 8. Relevance to pyorps GPU Implementation

### What pyorps V4 Already Does Well
- **Persistent cooperative kernel**: avoids kernel launch overhead (like ADDS)
- **Implicit raster graph**: no CSR overhead, predictable neighbor access
- **Custom atomic barrier**: correct approach for inter-block sync (validated by literature)
- **Large delta**: consistent with Lesnikov's finding that GPU needs much larger delta

### Potential Improvements from Literature

1. **Dynamic delta (from ADDS)**: Instead of fixed delta, monitor GPU utilization and adjust. Could help on heterogeneous cost rasters where optimal delta varies by region.

2. **Multi-bucket work queue (from ADDS)**: 32 circular FIFO buckets instead of single-bucket processing. Better work ordering without the overhead of full priority queue.

3. **MTB delegation pattern (from ADDS)**: Dedicate one block as manager to coordinate work distribution. Avoids contention on shared control buffers.

4. **Load-balanced partitioning (from Davidson)**: Less critical for raster (uniform degree) but could help for constrained SSSP where state space has irregular degree.

5. **Locality-based relaxation (from Safari)**: Each thread processes k-hop neighborhood instead of single cell. Could reduce number of synchronization rounds in persistent kernel. Natural fit for raster grids where k-hop = diamond pattern.

6. **Double-buffered flags (from Safari)**: Avoid race conditions on frontier flags without atomics. Clean alternative to atomic-based flag updates.

### What Probably Won't Help
- **parBucketHeap** (Berney): too complex, designed for dense high-diameter graphs, not sparse raster grids
- **Full CSR conversion**: raster graphs should remain implicit — CSR would waste memory and lose cache advantages
- **cuGraph/nvGRAPH**: already benchmarked by pyorps, 5-25x slower than custom kernel

---

## 9. Paper Reference Table

| # | Authors | Title | Venue | Year | Key Contribution |
|---|---------|-------|-------|------|-----------------|
| 1 | Davidson, Baxter, Garland, Owens | Work-Efficient Parallel GPU Methods for SSSP | IPDPS | 2014 | Workfront Sweep, Near-Far, Bucketing; load-balanced traversal |
| 2 | Wang, Fussell, Lin | A Fast Work-Efficient SSSP Algorithm for GPUs (ADDS) | PPoPP | 2021 | Dynamic delta, MTB delegation, async multi-bucket; **SOTA** |
| 3 | Berney, Iacono, Karsin, Sitchinava | A Parallel Priority Queue with Fast Updates for GPU | arXiv | 2023 | parBucketHeap; parallel Dijkstra for dense high-diameter graphs |
| 4 | Lesnikov, Chernoskutov | Performance analysis of Delta-stepping on CPU and GPU | CEUR-WS | 2016 | GPU needs 10x larger delta than CPU; delta sensitivity analysis |
| 5 | Safari, Ebnenasir | Locality-Based Relaxation for GPU Shortest Paths | TTCS/HAL | 2017 | k-hop DFS per thread; minimal kernel launches; road network focus |
