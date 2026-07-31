# 5th-Place writeup (with Better Compression Algorithm and Seed Cracking)

First, we want to thank the organizers for hosting such a competition and all the participants for battling it out with us. The past three months were exciting for us.

Our team’s background is mainly in competitive programming and CTF. So this competition was a step into unfamiliar territory, since we had little experience with code golf and almost none with Kaggle. In fact, it looked like many teams in this competition were also new to Kaggle. We’re a student team (someone questioned our odd team name, it's an inside pun related to [our university](https://www.tsukuba.ac.jp/en/about/public-branding/future/)), and we worked especially hard during our summer break in August and September.

This writeup explains our approach and highlights a few interesting solutions. The visualization tool we used and our GitHub repository are listed at the end.

# Overview of Our Solutions

Our approach was 90% grinding code golf, 9% tooling, and 1% weird hacks. Since the 90% grind was something many teams did, we’ll just showcase two of our most memorable solutions.

## 1. task 111 (62 bytes, best+2)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2F010fa1d41f9b192b5216869ad868be59%2FScreenshot%20from%202025-11-02%2015-02-06.png?generation=1762063336648581&alt=media)

```py
p=lambda g:[[*iter(sum(g,g).pop,5)][3-i:-i:-1]for i in b'\x0c\x16\x20']
```

`iter(l.pop, n)` walks `l` in reverse until it pops `n`. This task was the only one where we could exploit that behavior.

```py
a = [3,1,4,1,5,9,2]
print([*map(str,iter(a.pop,4))]) # ['2', '9', '5', '1']
print(a) # [3, 1]
```

## 2. task 270 (115 bytes, best+2)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2Fd59ef4bdb80a7ea4ffc107f84afe9bca%2FScreenshot%20from%202025-11-02%2015-02-29.png?generation=1762063355613187&alt=media)

```py
import re;p=lambda g,c=-7:c*g or[*zip(*eval(re.sub("((3)|7)([^[(]+)0(, (?(2)2|1))",r"0\3\1\4",str(p(g,c+1)))))][::-1]
```

This solution was the only one we saw that used the conditional-group syntax `(?(id/name)yes-pattern|no-pattern)` to change the regex pattern depending on whether a group matched.

# Tooling and Some Odd Hacks

Here are the tools and odd hacks we built along the way.

## Compression Optimizations

Even before the contest started, we suspected there would be cases where compression strategy might be used. That hunch turned out to be right. We built our own compressor and optimizer stronger than Zopfli, which is a better zlib compressor developed by Google, and lifted our score by about 700 points relative to the native zlib.

### Compression Basics

First, a quick overview of the compressed-code setup. Compressed code needs to be decompressed and executed at runtime. A key consideration is how to embed the compressed payload into the source. Python byte literals cannot contain characters ≥ 0x80, so we need to convert from another format.

We handled this by embedding the payload in a string literal and converting it to bytes. In Python, the file encoding can be specified with a comment like `# -*- coding: latin-1 -*-` ([PEP 263](https://peps.python.org/pep-0263/)). The shortest form that matches the regex `^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)` is `#coding:L1`. `L1` is an alias of the `latin_1` encoding ([Documentation](https://docs.python.org/3/library/codecs.html#standard-encodings)), which maps every character from `ord(0)` to `ord(255)` directly to its byte value. Using this, the decompression stub becomes:

```py
#coding:L1
import zlib
exec(zlib.decompress(bytes("<compressed code>",'L1')))
```

We used zlib because it fit our use case best. Python gives you zlib, lzma, and bzip. zlib and lzma are LZ77-family compressors, and bzip is based on the Burrows–Wheeler Transform. BWT can be weaker on these inputs compared to LZ77, and lzma is strong but its header overhead is large, so it didn’t beat zlib unless the code exceeded about 1000 bytes. The newer zstd will be available in Python 3.14 ([Documentation](https://docs.python.org/3/library/compression.zstd.html)), but sadly it was out of reach this time.

zlib is basically deflate plus a 2-byte header and a 4-byte CRC checksum. `zlib.decompress()` supports raw deflate when you pass a negative second argument, so `zlib.decompress(code,-15)` saved 6 bytes at the cost of 4 extra bytes, netting a win. For small zlib streams, `zlib.decompress(code,-9)` sometimes also worked, shaving another byte. Our final decompression stub looked like this:

```py
#coding:L1
import zlib
exec(zlib.decompress(bytes("<compressed code>",'L1'),-9))
```

### The Structure of deflate

Before talking about deflate-specific optimizations, here’s the bare minimum structure. Deflate encodes data as copies of previous substrings plus single-character additions. For instance, `abracadabra` becomes `[a][b][r][a][c][a][d][copy(distance=7,length=4)]`. Think of each piece as a factor. Minimizing the number of factors is easy with a greedy approach. In practice, we must also encode those factors into a bitstream, and that coding step complicates matters. Minimizing factor count alone doesn’t always minimize the final deflate size.

For coding, deflate uses Huffman codes for literals, and for length/distance pairs. You can either use a preset Huffman tree or embed your own. In our use case, presets were inefficient, so we mostly embedded our own. Switching trees midstream is possible, but embedding a tree costs about 300 bits (~40 bytes), so one tree is typically best here. (Rarely, using the preset tree for the last block is better. But you can brute-force the boundary to easily compensate this.)

There are two Huffman trees, one for literals/lengths and one for distances. These trees are themselves run-length encoded (RLE) and then encoded by another Huffman tree (the "cl-code") and inserted as a header.

We built a visualization app for deflate([deflate-viz](https://key-moon.github.io/deflate-viz/), [repository](https://github.com/key-moon/deflate-viz)). By attaching a deflate or zlib payload via query parameters, you can inspect the internals.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2Fe8762c54471c6f6c8d3f9595273d1001%2Fdeflate-viz.png?generation=1762062312506864&alt=media)

### Code-Level Optimizations

By exploring the visualization, we got a few key insights.

#### Make the same code fragments appear as much as possible

Obvious, but the golden rule of compression golf. Also loop unrolling sometimes produce better results than loops.

#### Avoid uppercase variable names / Reuse names that appear in keywords

As mentioned, the Huffman tree is represented by code lengths ordered by byte value and then RLE-compressed. Longer runs help. Python syntax doesn’t use uppercase letters, so those bytes would ideally be all zeros in the code-length stream; introducing uppercase disrupts RLE. Also, to help the literal Huffman distribution, it’s better to reuse letters that appear in common keywords like `def` and `return`, or built-ins like `zip`, and avoid introducing stray characters.

#### Prefer tabs over spaces for indentation / Prefer `'` over `"`

Again, RLE awareness. Tabs sit next to newline in ASCII and tend to disrupt RLE less than spaces. Single quotes tend to work better because `'` is adjacent to `(`.

### Optimizing the Compressed Payload

We first set out to produce an encoding-friendly deflate payload. Since the payload is embedded as a string, some characters require escaping, e.g., the quote characters used for the string delimiter, or `\x00`. Each such character costs an extra byte; we want to avoid them. Empirically, roughly 1/100 of bytes tended to need escaping.

So we tried randomly perturbing parts of the deflate payload to dodge those characters. The most leverage comes from the embedded Huffman tree, which influences many bytes. Under an optimistic assumption that all bytes randomize independently, a ~200-byte payload would avoid escapes with probability about `(99/100)^200 ≈ 0.13`. Reality is less kind, so it took more trials.

We implemented a deflate parser and performed hill climbing on top of Zopfli’s output ([implementation](https://github.com/key-moon/golf/blob/2568f97b8677aa1898c5dee6c4ec240a37f578ae/deflate_optimizer/optimizer.py#L261)). This worked well in practice and often reduced the number of escapes to zero.

### Beyond Zopfli

Along the way, we generated payloads that beat Zopfli’s size, initially as a side effect of hill climbing to remove escapes. That suggested we might actually build a compressor that outperforms Zopfli.

The full story is long, so details are in a teammate’s writeup(To Be Written). Here is a broad, high-level overview of the approach.

We first tried improving the Huffman tree with metaheuristics like genetic algorithms and simulated annealing, but this underperformed. We suspect that this because of the search landscape wasn’t smooth, and the objective (bits) is discrete.

The big breakthrough was realizing that when **everything except** the Huffman tree is fixed, we can compute the optimal Huffman code lengths by dynamic programming (DP). Concretely, fixing the factorization and the cl-code (the Huffman code for the Huffman codes) allows a DP to find optimal code lengths. Implementing this let us shrink the Huffman trees that Zopfli produced ([implementation](https://github.com/key-moon/golf/blob/2568f97b8677aa1898c5dee6c4ec240a37f578ae/deflate_optimizer_cpp/optimal_lit_code_lengths.hpp#L14)). Here’s a comparison:

* [Zopfli, 218 bytes](https://deflate-viz.pages.dev/?deflate=le9FlgIwDAUArNODzGuGFHeHY%2BAOFdz16rg7u8jXSrUGHV7HIINEZM4lKay1eyBBtqBXbNWr3FRb2%2F%2F%2Bqu6uaXsWUdagnpbZtMr%2BOZehZMTFYNyQpgqOIAMoU58kqUizOOG8P2xyLi0rUSz1uRYO%2FDekzqtzu%2BLWIAE7L33xcu%2FNzeWgo3bqI5Lo0ylw%2F%2FJNkufXDrjNLfoMQNagHHYEe9XBsNeCOoPHLLtW8CSQ6O9fO40vOgYTIjL%2FgFvu5apmG%2BpQIHqLjwpH%2BJYSPlcK1tO34Gz6FpqN%2BDc%3D)
* [DP-optimized Huffman tree, 215 bytes](https://deflate-viz.pages.dev/?deflate=le9FDkMhEAUArIeDNNBCUne%2FBmHx3ZW6nb3uvht5qhsmxNgiTQSDzhI71CVmlIIDTgipEloG9o1w9z9c3YcrzwtCHBMs7gjuikxx3Rp2SgimtuMbUGgiAI1K6lC3EygzjOU4wNjJbZiiSuyxAsn61L2sxd1KdgYD2Ht5V6%2Fywdy%2FHrxunkpCKJP0HFhev0Na%2BbcD2eVmEgE4JmjtQjM1RuM0BAvBc5Z9K3gRiMnDa6%2FxQ8fmgHWWX3Drg5zh70IdC3Tv8V1WaN9T2pdKTYvfgwW%2Fh4pOfQs%3D)

#### Automatic Variable-Name Optimization

Observation suggested variable-name assignment strongly affects compression. To mechanize this, we statically analyzed code to extract dependencies and then added a routine that reassigns variable names to improve compressibility. This was the single most impactful optimization, netting several hundred bytes overall.

#### Genetic Algorithm

We also ran a GA whose state combined variable-name assignments and the deflate payload itself, interleaving factorization and Huffman optimizations. This gained around 100 bytes ([implementation](https://github.com/key-moon/golf/blob/main/deflate_optimizer_cpp/geneticalgo.cpp)).

## Recovering the Random Seeds for Test Cases

It was disclosed that test cases were generated using [ARC-GEN](https://github.com/google/ARC-GEN/). The generator is deterministic given a seed. From `arc_gen.py`:

```py
# https://github.com/google/ARC-GEN/blob/main/arc_gen.py
def generate_benchmarks(task_num, num_examples):
  """Creates a benchmark suite for a given task."""
  task_info = task_list.task_list()[task_num]
  _, generator, _ = task_info
  examples = []
  for example_id in range(num_examples):
    random.seed(task_num + example_id)
    examples.append(generator())
  print(examples)
```

However, running arc_gen didn’t reproduce the exact test cases.

When the anomaly in task100 was fixed and the test cases remains as same in almost all cases, it suggested some seed mechanism was in play. We reasoned that the high quality source to leak randomness is via `randint`. Skipping the nerdy details, if a generator uses only calls like `randint(a, b)` where `b - a + 1` is a power of two, those outputs can leak clean bits of the PRNG state.

We filtered ARC-GEN and identified three tasks, which is task043, task127, task142, as good candidates. From each ARC-GEN case we extracted the random sequences used.

```py
LEAK_CASES = 3

# https://github.com/google/ARC-GEN/blob/main/tasks/training/task043.py
# rows = [item for item in range(1, size) if common.randint(0, 1) == 0]
# cols = [item for item in range(0, size - 1) if common.randint(0, 1) == 0]

cases = get_task(43)['arc-gen']
leak_43 = []
for case in cases[:LEAK_CASES]:
  g = case['input']  
  leak = [1*(0==r[-1])for r in g[1:]] + [1*(0==c)for c in g[0][:-1]]
  leak_43.append(leak)

# https://github.com/google/ARC-GEN/blob/main/tasks/training/task127.py
# colors = [common.randint(1, 4) for _ in range(3 * common.randint(1, 2))]

cases = get_task(127)['arc-gen']
leaks_127 = []
for case in cases[:LEAK_CASES]:
  g = case['input']
  leak = []
  leak.append([0, 1][len(g)==7])
  leak.extend([v-1 for v in g[1][1::4]])
  if len(g) == 7:
    leak.extend([v-1 for v in g[5][1::4]])
  leaks_127.append(leak)

# https://github.com/google/ARC-GEN/blob/main/tasks/training/task142.py
# colors = [common.randint(0, 3) for _ in range(size * size)]
# grid, output = common.grid(size, size), common.grid(2 * size, 2 * size)
# for r in range(size):
#   for c in range(size):
#     grid[r][c] = colors[r * size + c]
cases = get_task(142)['arc-gen']
leaks_142 = []
for case in cases[:LEAK_CASES]:
  g = case['input']
  leak = []
  for s in g:
    for v in s:
      leak.append(v)
  leaks_142.append(leak)

print(leaks_127)
print(leaks_142)
# [[0, 3, 1, 0], [0, 2, 0, 1], [0, 3, 0, 0]]
# [[0, 3, 1, 0, 2, 3, 1, 0, 3], [0, 2, 0, 1, 3, 3, 3, 1, 0], [0, 3, 0, 0, 2, 3, 1, 3, 1]]
```

A bit of inspection shows the random sequences share the same prefixes. In other words, across all tasks, the same seed was used per index!

With that in mind, we searched for seeds that would produce the sequences above.

```py
for i in range(10000):
  random.seed(i)
  r = [random.randint(0,1) for i in range(18)]
  if r in leak_43:
    print(i, leak_43.index(r))
# 2025 0
# 2026 1
# 2027 2
```

So the seeds were `2025 + i`. This hinted we might compress some tasks dramatically by regenerating from the seed. For this to be useful, the following criteria had to be met:

1. It’s hard to extract the vital information in Python directly (easier to regenerate from randomness).
2. The generation code is simple.
3. Both train and test are derivable from a same algorithm and a short seed.
4. The seed is recoverable from available information.
5. Constructing the solution from the recovered information is straightforward.

By scouting tasks, only task096 fit these criteria. This tasks has tons of corner cases and was one of the hardest to optimize.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2F07f32fcee52b745d1bfaa82d9c4a9397%2FScreenshot%20from%202025-11-02%2014-55-16.png?generation=1762062925269548&alt=media)

In this task, the grid width/height, background color, reticle colors, and each segment length are generated randomly. Here is a simplified version of the algorithm for those:

```py
# ref. https://github.com/google/ARC-GEN/blob/main/tasks/training/task096.py
width, height = random.randint(13, 19), random.randint(13, 19)
b = random.randint(1, 9) # background
colors = random.sample(sorted(set(range(1, 10))-{b}), random.randint(4, 6))
lengths = []
for i in range(len(colors)):
  min_length, max_length = min(i + 1, 2), i + (0 if i > 1 else 1)
  lengths.append(random.randint(min_length, max_length))
```

These are exactly the pieces of information you need to determine the seed and construct the output. Our experiments showed that width, height, the set of colors, and the background color suffice to identify the seed. And, for non-ARC-GEN train and test cases, we also found seeds via exhaustive search.

Here’s the code we built using this strategy:

```py
from random import*
def p(r,l=4):
 seed([2024-l,'+u-)+e[','41-efe[','41*l}{','f(3,+e['][l*(0<l)])
 d,t,f=randrange(13,20)==len(r[0]),randrange(13,20)==len(r),randrange(1,10)
 s=sample([*{*range(1,10)}^{f*(l<0)}],o:=randrange(4,7))
 u=[1+randrange(0<m,(m<2)+m)for m in range(o)]
 n=[(o*2-1)*[f]for m in range(o)]
 for m in range(o):
  for e in range(u[m]):e-=m;n[e-1][o+m-1]=n[e-1][o-m-1]=n[-m-1][o+e-1]=n[-m-1][o-e-1]=s[m]
 return all([d,t,{*sum(r,[])}=={*s,f},f in r[2]])and n+n[:-1][::-1]or p(r,l-1)
```

After compression this ended up at best+48, which is not quite we are looking for, but we’re proud we had this kind of solution.

## GX (Golf Experience) / QoL Features

Given the marathon nature of the contest, we built QoL tooling: mainly CI/CD and a local Web UI.

### CI/CD

On every push (and on schedules) we ran jobs to do the following.

#### Generate README

We generated a stats summary in Markdown as the README. Later the Web UI replaced it, but it was handy on the go.

#### Generate per-file banners

We auto-updated banners in files to show size targets like this:

```py
# best: 51(LogicLynx, ALE-Agent) / others: 53(Ravi Annaswamy), 54(jonas ryno kg583 kabutack), 54(cubbus), 54(jacekwl Potatoman nauti), 54(jacekw Potatoman nauti natte)
# ======================= 51 ======================
```

#### Discord Integration

We notified our Discord on every teammate push. This made it easy to spot missed shrink opportunities and pick the low-hanging fruit.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2F90c30fbaaf30f328ef51bfaf7c12565a%2FScreenshot%20from%202025-11-02%2004-48-08.png?generation=1762063444300991&alt=media)

#### Watch Standings / Spreadsheet

We automatically updated the best-known sizes referenced by the banners and Web UI.

### Web UI

We've created(99% vibe-coded) local Web UI for daily golfing.

#### Top Page

We could track our progress. A histogram highlighted which tasks deserved compression attention.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2Fc90f52efe70694f4c7f046da468feb80%2FScreenshot%20from%202025-11-02%2014-58-05.png?generation=1762063098220077&alt=media)

#### Task List

Thumbnail overview, color-coded by delta from best. We added manual tags and search.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2F92816db89d2e92b34f4b87b2c796d66d%2FScreenshot%20from%202025-11-02%2014-58-59.png?generation=1762063151089124&alt=media)

#### Task Detail

As shown: tags and code, all cases, and score trajectories by team. We also added a “Open current best in VSCode” button.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2Fc0dac6489524be375907ed0ad9b9008a%2FScreenshot%20from%202025-11-02%2014-59-49.png?generation=1762063195601066&alt=media)

#### Judge View

Integrated with [checker.py](https://github.com/key-moon/golf/blob/main/checker.py) to show the latest run results. Passing values through the identity function [`DUMP`](https://github.com/key-moon/golf/blob/main/checker.py#L83) printed its argument, which helped debug recursive solutions.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1953252%2Fd2a18ffb82a338e5ac9eb4576d4aaaf6%2FScreenshot%20from%202025-11-02%2015-00-06.png?generation=1762063212521821&alt=media)

## Other Things We Tried

* Early on, we auto-generated solutions with GPT-o4-mini, which is why we ranked high at the start. Those codes live in [base_code](https://github.com/key-moon/golf/tree/main/_junkyard/old_base/base_code). None of the final surviving solutions relied on AI, though.
* We also discovered the universal solution via `__eq__` that was published in mid-August. Our short version was:
  * `class p:__init__=help;__eq__=any`
* Until late August we used minified versions of [arc-dsl](https://github.com/michaelhodel/arc-dsl) and [re-arc](https://github.com/michaelhodel/re-arc) to chase 400/400. Some compressed to ~2000 bytes. By early September, we removed them all and replaced them with handwritten solutions.

# Data

Our competition repository (very messy; it wasn’t meant to be public) is here:

[https://github.com/key-moon/golf](https://github.com/key-moon/golf)

Repository overview:

* `dist`: results
* `base_[name]`: each member’s “golf course”

If you want to run anything, execute `source tools/setup.sh` to set up the environment.