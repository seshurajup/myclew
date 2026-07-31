# Santa 2025 5th-place solution writeup

## Algorithm overview

- Generate initial solutions with regular patterns
	- Each pattern consists of tiled pairs of trees
- Improve solutions
	- Shrink the square boundary just a bit, and then run SA to minimize overlap
	- Uses a **physics simulator** to quickly reach a local minimum, which is written just for this competition
	- Tries to reuse N-1 and N+1 solutions to build N solution in order to:
		- escape from a local optimum
		- propagate "good patterns"

## Source code

[Link to the GitHub repo](https://github.com/saharan/santa-2025)

## Generate initial solutions

In initial solution generation, I tile pairs of trees to get good regular patterns.

There are two modes in tiling:

- mode 0:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30809874%2F13efcd934bbacfd292a08aa2ae2b804d%2F064i.png?generation=1770487393387483&alt=media)
- mode 1:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30809874%2F8449ea7bb3183188448aad4a8654c6c5%2F065i.png?generation=1770487408701694&alt=media)

Mode 0 has higher density, but less flexibility, while mode 1 has less density, but more flexibility. I used different modes for different Ns. However, apparently **this was the biggest cause of the defeat** in the very last of the competition; actually mode 0 was way better in the most cases :(

I ran SA for the both modes to get locally optimal solutions, and then used the better one as the initial solution.

See `InitialSolver` in `Main.cpp` for more details and the implementation.

## Physics-powered searching

As some people was discussing it in the official forum, physics simulation can be useful for this kind of problem.

However, I didn't think most open-source physics engines can be used directly for two reasons:

1. Precision.
	- The main purpose of those physics engines are to get realistic behavior, not to get the snapshot of a scene with an extreme precision.
	- Small overlaps (>> EPS) are usually tolerated to avoid jittering.
1. Collision handling.
	- Most physics engines cannot handle concave shapes directly.
		- We have to decompose them into convex polygons.
	- And **they stuck very often when largely overlapped!**
		- You know physical objects in games sometimes go crazy...

So I made my own for the competition. Features:

1. Precision.
	- Capable of handling objects with an extreme precision.
		- I was using `double` at first, but changed to `float` to double the SIMD performance.
	- Never tolerate overlaps.
		- Requires a tiny separation (`4e-7`) between objects. The value is chosen so that:
			- it won't cause floating-point issues (well it does sometimes to be honest...)
			- it little affects the score
	- Uses a nonlinear solver to speedup the convergence.
		- Corresponds to the "Nonlinear Gauss-Seidel solver" in Box2D.
2. Collision handling.
	- Made a custom collision handler so that **trees never get stuck overlapping**.
		- I think PBD (position-based dynamics) helped it.
		- Also added an auxiliary shape for each tree after convex decomposition.

With it, we can project a solution to the local optimum (i.e. the state with (locally) lowest energy w.r.t. overlaps). We call this projection **"diffusion"** since it diffuses the overlap amounts of trees by slightly translating and rotating them.

I believe it greatly improved the local search efficiency!

### Convex decomposition result

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30809874%2Fb47ec470bdb68e170c7d3d37990b7e41%2Fall.png?generation=1770557137810001&alt=media)

## Simulated annealing

The main part of the optimization.

It works roughly as follows:

1. Initialize with a feasible (no overlap) solution.
1. Shrink the field a bit by moving the boundary walls.
1. Diffuse the field and start the loop:
	1. Try an SA transition.
	1. If accepted, sometimes try diffusing the entire field.
	1. If there is no longer overlaps, it means that we found a better solution!

### Transitions

To maximize the acceptance rate, **trees with larger surrounding spaces** are more likely to be moved. In addition, small groups (each contains 8 to 16 neighboring trees) are precomputed to speedup the diffusion process.

#### Transition 1 (90%)

1. Pick two or three neighboring trees that belong to the same group.
1. Freeze the other trees (i.e. treat them as walls) and remove the picked trees.
1. Randomly insert the trees and hill-climb a bit.
1. Unfreeze the trees in the group and diffuse a bit.
	- This is faster than diffusing the entire field, especially when N is large.

#### Transition 2 (10%)

1. Pick a random group. Freeze all trees outside the group.
1. Perturbate 1-4 trees in the group.
1. Diffuse a bit.

### Temperature

I used **parallel tempering** for SA, which enables us to run it indefinitely. Eight to twelve systems are used, depending on N and the optimization mode.

#### SA Search in a higher temperature

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30809874%2F3148cc8ddb01e267a99e1549f8e956ce%2Fsa-hightemp.gif?generation=1770557075088317&alt=media)

#### SA Search in a lower temperature

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30809874%2F9d86e4f8e9c07fd25f0791d94090dd76%2Fsa-lowtemp.gif?generation=1770557099463592&alt=media)

## Optimization modes

There are several optimization modes that determine how the solutions will be optimized.

### Use N-1 or N+1 solution

In this mode, we first start with the N-1 or N+1 best solution, and then insert or delete a random tree to get a solution with N trees.

Then we start optimizing the solution normally. This process is helpful to **propagate useful patterns** to the neighboring solutions.

### Improve the best solution

In this mode, we just try to improve the current best solution for each N.

### Exploit symmetry

In this mode, we only search for symmetric solutions. Symmetric constraints are applied during the physics simulation.

Since I use PBD for the physics simulation, all I have to do is to adjust the trees' positions during the solver iterations.

## Featured solutions

### N = 100, 144, 196

Characteristic examples for some \\(N=(2k)^2\\). Looks pretty but sadly they were still far from optimal (see other top writeups).

![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/100.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/144.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/196.png)

### N = 172

This case has **the lowest score among all solutions**.

![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/172.png)

This is a rare case where the mode 1 configuration actually beats the mode 0 configuration.

## Final solution plots

![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/001.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/002.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/003.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/004.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/005.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/006.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/007.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/008.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/009.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/010.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/011.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/012.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/013.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/014.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/015.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/016.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/017.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/018.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/019.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/020.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/021.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/022.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/023.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/024.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/025.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/026.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/027.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/028.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/029.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/030.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/031.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/032.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/033.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/034.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/035.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/036.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/037.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/038.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/039.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/040.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/041.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/042.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/043.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/044.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/045.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/046.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/047.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/048.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/049.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/050.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/051.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/052.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/053.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/054.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/055.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/056.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/057.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/058.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/059.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/060.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/061.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/062.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/063.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/064.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/065.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/066.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/067.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/068.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/069.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/070.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/071.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/072.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/073.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/074.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/075.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/076.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/077.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/078.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/079.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/080.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/081.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/082.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/083.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/084.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/085.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/086.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/087.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/088.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/089.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/090.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/091.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/092.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/093.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/094.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/095.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/096.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/097.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/098.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/099.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/100.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/101.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/102.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/103.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/104.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/105.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/106.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/107.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/108.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/109.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/110.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/111.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/112.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/113.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/114.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/115.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/116.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/117.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/118.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/119.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/120.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/121.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/122.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/123.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/124.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/125.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/126.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/127.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/128.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/129.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/130.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/131.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/132.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/133.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/134.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/135.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/136.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/137.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/138.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/139.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/140.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/141.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/142.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/143.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/144.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/145.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/146.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/147.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/148.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/149.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/150.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/151.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/152.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/153.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/154.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/155.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/156.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/157.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/158.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/159.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/160.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/161.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/162.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/163.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/164.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/165.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/166.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/167.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/168.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/169.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/170.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/171.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/172.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/173.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/174.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/175.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/176.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/177.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/178.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/179.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/180.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/181.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/182.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/183.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/184.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/185.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/186.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/187.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/188.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/189.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/190.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/191.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/192.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/193.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/194.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/195.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/196.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/197.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/198.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/199.png)  
![](https://raw.githubusercontent.com/saharan/santa-2025/refs/heads/main/imgs/200.png)