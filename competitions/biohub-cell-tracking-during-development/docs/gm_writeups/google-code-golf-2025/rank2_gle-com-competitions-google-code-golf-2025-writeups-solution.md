# 2nd place solution - jailctf merger

# Summary
Our team was formed from 5 CTFers involved in the organization of jailCTF, one of whom is a seasoned code golfer. We focused on getting tooling for compression and easy collaboration very early on the competition, hoping that a team effort could make up for the lack of golfing experience the other four of us had. In the last month, we wrote a fuzzer to get some easy bytes on some tasks, which ended up finding a cool trick and helped save 100+ bytes in the final stretch.

The work was approximately split as such:
- **lydxn** did most of the heavy code golfing, finding the majority of the tricks in our solutions
- **quasar098** worked on much of the tooling
- **oh_word** focused on compression and compression based tasks
- **hellopir2** did a lot of optimization all around
- **Quasar-0147** was mostly inactive

## Submission and Tooling
All of the code and tooling that we used can be found in the repositories [here](https://github.com/orgs/jailctf-merger-goog-golf/repositories). Specifically, charbrute and var_genetic can be found in [this repo](https://github.com/jailctf-merger-goog-golf/compression). The [submission.zip](https://github.com/jailctf-merger-goog-golf/golf/blob/master/submission.zip) file was our final submission. The spreadsheet we used as a central hub for information can be found [here](https://docs.google.com/spreadsheets/d/1mjz6tYb2caNHD6-PktIrHBdiJda5TjXRqKcUaiTQc4E/edit).

# Charbrute
We called the fuzzer charbrute, because what it does is randomly mutate the code at the character level (deletions, substitutions, insertions). A so-called "astbrute" was proposed but never implemented due to high complexity and low perceived gains.

The fuzzer was originally pretty bad, and only handled deletions and substitutions. Over time, we added a few improvements, the major ones being:
- Handling insertions as a replacement for substitutions
- Adding a chunk mover which moved around chunks of code (this one didn't really give good results, maybe moving it to before the other mutations would've helped)
- Forcing the spot mutations to be centralized around a hotspot with a probability distribution
- Using pypy, since pypy3.11 came out recently. This creates a few minor differences such as set ordering and float caching but these can be manually filtered or blacklisted.

It mostly found minor saves and optimizations in our code, but the occasional large save was found. It very quickly got saturated for our code dataset, and so we were forced to keep adding new features or slight modifications to keep its productivity.

Examples of some saves that it found are below:
```py
# task 131, -1b
p=lambda g:exec("c=2;g[:]=zip(*([j for*j,in g if(c:=c-(s:=max(j))*(c>0))|s]+[[8]*9]+[[0]*9]*99)[len(g)-1::-1]);"*4)or g
p=lambda g:exec("c=2;g[:]=zip(*([j for*j,in g if(c:=c-(s:=max(j))*(c>0))|s]+[[8]*9]+[g[0]]*99)[len(g)-1::-1]);"*4)or g
# human optimization, -1b
p=lambda g:exec("c=2;g[:]=zip(*([j for*j,in g if(c:=c-(s:=max(j))*(c>0))|s]+[[8]*9]+g[:1]*99)[len(g)-1::-1]);"*4)or g

# task 125, -3b
p=lambda g,i=87,q=8:g*-i or[[[12%c,~q%7//c*8,-(q&(q:=c)%3)%5][i//42]or c for c in g]for g[::-1]in zip(*p(g,i-1))]
p=lambda g,i=87,q=8:g*-i or[[[12%c,q//c*c,-(q&(q:=c)%3)%5][i//42]or c for c in g]for g[::-1]in zip(*p(g,i-1))]

# task 196, -3b
p=lambda g,i=23:g*-i or[[q:=[c|4-c&6%~q,c|-c&~q%3,c%4][i//8]for c in g]for g[::-1]in zip(*p(g,i-1))if(q:=8)]
p=lambda g,i=23:g*-i or[[q:=[c|8-c&q,c|-c&~q%3,c%4][i//8]for c in g]for g[::-1]in zip(*p(g,i-1))if(q:=8)]
```

The most notable trick we fuzzed is what I call "pop rotation".

For some context, shortly after we matched the public best on task015, which was 93 bytes at the time, it looked like this:
```py
p=lambda g,i=-7:g*i or p([[r.pop()%9|36%(6^-[0,*r][i//7])%13for _ in g]for*r,in zip(*g)],i+1)
```

Note the unique inside-out recursion structure with the `p` call on the outside (which was rarely done because putting the recursion on the inside saved 1 whitespace character), and the squareness of all the input grids.

The fuzzer found this:
```py
p=lambda g,i=-7:g*i or p([[r.pop()%9|36%(6^-[0,*r][i//7])%13for r in g]for*r,in zip(g)],i+1)
```

And it turns out you can just completely remove the `zip()` for the current known best of **87 bytes**:
```py
p=lambda g,i=-7:g*i or p([[r.pop()%9|36%(6^-[0,*r][i//7])%13for r in g]for*r,in g],i+1)
```

This is pop rotation, where you perform a 90 degree counterclockwise rotation of a grid with just `.pop()`!

For comparison, let's see the other "rotate a grid 90 degrees in a list comprehension" formats:
```py
# square grid:
p=lambda g:[[r.pop()for r in g]for r in g]
p=lambda g:[[r.pop()for _ in g]for*r,in zip(*g)]
p=lambda g:[[c for c in g]for g[::-1]in zip(*g)]

# rectangular grid:
p=lambda g:[[r.pop()for r in g]for r in g[0]*1]
p=lambda g:[[r.pop()for _ in r*1]for*r,in zip(*g)]
p=lambda g:[[c for c in r]for r in zip(*g)][::-1]
```

Of course, it has its own advantages and disadvantages compared to the other typical rotation formats, which is why it is not able to be used everywhere.

In the end, we applied this on these tasks:

![thing](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28145366%2Fbcfb8d20f5e6cd698bd948d6ef078a3d%2Fimage.png?generation=1763018902322784&alt=media)

# Compression
For compression, we utilized a variable rename optimization script, similar to the public ones.

There are two key differences though:
1. Stripping the header and checksum and using `wbits=-9` in the decompression saves a few bytes.
2. Using a genetic algorithm (implemented by gemini 😅) to optimize for raw compressed length, compared to naive brute force.

Overall, these optimizations saved a few bytes on each compressed task that we had, which was down to just 24 by the end of the competition.

Our initial approach to the longer tasks was to get an LLM to generate an initial working solution by prompting it with the generation code, general code golfing tips, and some human ideas for the specific task at hand, and then afterwards optimizing the code manually for both compressibility and length. However, most of the llm solutions we golfed got obsoleted by either public notebooks or with a solution rewrite.

Other than that, we had a few main strategies, either to rewrite the code for compressibility/brevity or to repeat/reuse code to do different subtasks. We abused the second strategy quite a lot more than the first. In fact, out every compressed diamond, we own the two highest compression ratios of ~2.91 on our task044 diamond and ~2.46 on our task173 diamond.

Some examples of compression optimizations we found are below:
```py
# "for r in " substring repeating during rotation
[r for*r,in zip(*g)][::-1]
[[*r]for r in zip(*g)][::-1]

# "g[y]" substring repeating during len checks
len(g[0])
len(g[y])

# "_ in range(len(_))" substring repeating during bounds checks
len(g)>y>-1<x<len(g[y])
y in range(len(g))!=x in range(len(g[y]))

# optimizing the huffman tree
x>y
x<y

# renaming variables to make your code more readable
[[c for c in r]for r in g]
[[g for g in g]for g in g]

# adding whitespace in the correct places
}for i in range(r)
} for i in range(r)
```

Note that these tips are not always positive changes! Mileage varies depending on the contents of the rest of the code. As such, optimizing for compressed length is a lot of trial and error (and pattern recognition), and when each compression attempt takes a couple minutes to brute force an optimized variable renamed version to check if it is shorter, it becomes very annoying.

However, the one benefit of compression golfing is that since no one started out particularly experienced at this, our team wasn't at a disadvantage with respect to this golfing skill. But it is quite annoying and very easy to waste a lot of time on if you don't approach the task correctly. Our team had many attempted rewrites that turned out to be a lot longer when compressed compared to the existing solutions, even if they were potentially shorter in terms of uncompressed length.

# Web UI

We created a centralized web server(s) for golfing and storing solutions, which the other top two teams did not do. This involved a flask server and a websocket server.

We did not have a git repository storing all of our solutions because that is too annoying to handle.

Features of the web UI + websocket server that promoted collaboration and efficiency:
- Store solutions and annotations (i.e. per-task notes)
- Real time updates to both solutions and annotations (using websockets)
- Basic authentication to prevent intruders
- Copy test cases button for easy regex manufacturing in regex101 or other
- View test cases button to see how the manually written test cases are wrong compared to the generated ones
- Gen code button to view test case generation code per task
- 20s timeout changeable to 90s timeout to prevent server strain by default
- Two-way integration with our central spreadsheet (google apps script) for easier analysis.
- Random buttons for golfable task finding
- Other tools integration

# pysearch

Like several other teams, we used @lynn's [pysearch](https://github.com/lynn/pysearch) tool to brute force arithmetic expressions.
A good example of this is task006, where we found an optimal 13-byte expression which isn't possible to find by hand:

```py
# 43b (gold)
p=lambda g:[[109//c&12%c+7for c in g[0]]]*3
```

pysearch tends to work well up to length 13/14 until it starts to become too slow. We tried to utilize pysearch as much as possible on any expressions below or around that length. In some cases when the expressions is too long to brute force, we can also perform a "partial" pysearch on parts of the expression.

After running pysearch on several tasks, we realized that specifying `INPUTS` and `GOAL` fields manually became extremely annoying and started generating the parameters programtically. Our gen script for task293 looked like this:

```py
from utils import pysearch
import json
import re

task_num = 293

with open(f'../data/training/task{task_num:03d}.json', 'r') as f:
    data = json.load(f)

testcases = data['train'] + data['test'] + data['arc-gen']

search = set()
for _, data in enumerate(testcases):
    g = data['input']

    for r in g:
        for c in g[0]:
            b = +(c in r)
            x = r[0]
            ans = c-c*b or x or c
            search.add((c, b, x, ans))

pysearch(search)
```

Based on the testcases, we fill a set called `search` containing tuples in the format `(input_1, input_2, ..., input_n, goal)` and pass that into a "utils" script that outputs Rust code to plug into `params.rs`:

```py
def pysearch(search, min_uses=1, max_uses=255, varnames=()):
    values = list(map(list, zip(*search)))

    big_template = 'pub const INPUTS: &[Input] = &[%s\n];'

    template = '''
    Input {
        name: "%s",
        vec: &%r,
        min_uses: %d,
        max_uses: %d,
    },'''

    output = ''
    for i, value in enumerate(values[:-1]):
        if not varnames:
            c = chr(i + 97)
        else:
            c = varnames.pop(0)
        output += template % (c, value, min_uses, max_uses)
    print(big_template % output)
    print('pub const GOAL: &[Num] = &%r;' % values[-1])
```

`c-c*b or x or c` was the formula we found by hand, while the shortest one is actually `b**c*x or c`. We basically reused this template at every oppurtunity for easy pysearch gains (which was a lot of tasks!). This was especially helpful for the "grid rotation" tasks, e.g. task002 / task243.

# Other Tooling

One useful tool we found online was called [infgen](https://github.com/madler/infgen/blob/1e36c3da205ce3da7df1abd1f1c765a3cf49b267/infgen.c) and it
allowed us to debug raw deflate streams, seeing both how many bytes each zlib mini-segment used and also how the substitutions were taking place in
ways that we might not be able to intuitively see.

Another utility we made was [autogolf](https://github.com/jailctf-merger-goog-golf/compression/blob/main/options/autogolf.py), which made sure
we didn't do any silly things with excess spacing or parenthesis and such. This tool was a heavy modification of `ast.unparse`, but with golfing in mind.

We integrated both of these to the golf server as buttons to allow us to easily use them.

# Interesting Tasks

## task048:
This was the only task where we hardcoded every test case using `hash()`, @lydxn made a blog post about task048 [here](https://lydxn.github.io/posts/fun-with-magic-hashes-in-python/). This trick could also be applied to task319 to find the right color.

## task387:
This task has you surround the nonzero colors with the other nonzero color, as well as connecting the pixels with an alternating strip of gray and black pixels.

![alt text goes here](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28145366%2Fd7649c9c7ac2057c667a1d8c75fca8d2%2Fimage-1.png?generation=1763018929275667&alt=media)

Looking at the other teams' solutions, we were the only team to use regex to solve the task, with it beating out the non-regex solutions of all teams but Code Golf International. So, we thought it would be interesting to explain the regex approach we came up with.

Firstly, notice that this task, like 99% of the tasks, can be greatly simplified if you only deal with one orientation at a time and then rotate. But for this task, the question of how you handle each orientation is not so easily answered.

The first thought might be to do something as follows, where you complete one 3x3 and one edge at a time:
```
___    ___     BBB     ___
_A______B_ --> BAB5_5_5_B_
___    ___     BBB     ___
```
(i disagree with markdown formatting in these code blocks, it looks ugly now)

However, what's interesting is that this proposal actually fills 2 3x3s and 2 edges on the initial grid! This gives a new idea: since each 3x3 and edge is going to be matched twice in the full rotation, we should try to fill in only half of an edge and half of a corner on each passthrough.

The edges are more apparent in how to approach than the 3x3s. Rather than filling the whole space between the two dots, you only need to fill it halfway. This turns our first transformation into something similar to the below:
```
___    ___     BBB     ___
_A______B_ --> BAB5_5___B_
___    ___     BBB     ___
```

The 3x3s are a little trickier in the approach. If we stare at it a little, we can see that we have a few options in which pair of opposite corners and which pair of opposite edges we want to fill in on the first pass.

```
proposal 1
___    ___     B__     ___
_A______B_ --> BAB5_5___B_
___    ___     __B     ___

proposal 2
___    ___     BB_     ___
_A______B_ --> _A_5_5___B_
___    ___     _BB     ___

proposal 3
___    ___     __B     ___
_A______B_ --> BAB5_5___B_
___    ___     B__     ___

proposal 4
___    ___     _BB     ___
_A______B_ --> _A_5_5___B_
___    ___     BB_     ___

```

Hold on, aren't we doing this in regex? Before we decide which proposal is the best, we need to figure out how to implement this in regex.

So the basic idea we had to implement this was like this:
```
[(stuff)___    ___
[(stuff)_A______B_
[(stuff)___    ___
```
Notice how since the rectangular frame is the only thing on the grid, all of the `(stuff)` groups should be equal. This fixes the issue that we don't know the distance from `A` to `B` which would otherwise make drawing the half dotted line from `A` to `B` quite annoying when trying to access the row below. Instead, if we capture `(stuff)`, we can just go to the next line by doing something like `[^[]+` and then matching `[(stuff)` to find the cells below `A`.

From here, there's not that large of a difference between the methods, so I picked the first one because it was the easiest to implement and probably slightly shorter. After a bit of golfing, we arrive at this 201 byte regex solution:
```py
import re
p=lambda g:[g:=eval(re.sub(r'([^)]*)0([^(]*\1)0(, ([^0]), )0,([^)]*)(?!0|\4)((\d).*?\1.{6})0',r'\1\7\2\7\3\7,*[5,0,5][:(c:=1|len([\5])//3)]+[\5][c:],\6\7',f'{*zip(*g),}'))[::-1]for _ in g][3]
```

Let's break down how this works.

Firstly, the `([^)]*)0` captures all characters between a `(` and a `0` on a single row. This means `\1` is the `(stuff)` that we wanted to match from earlier. The `0` is consumed because it will be replaced with color `B` in the replacement.

Then, the `([^(]\1)` captures all characters until the `0` below the previously consumed `0`. This `0` is again consumed because it will be replaced with the color `B`. Following this `0`, we expect `, A, ` in the row, which is matched with `(, ([^0]), )`, making `A` be group `\4`. The following `0,` is consumed, again to be replaced with `B,` later.

After this `0`, we expect the `B` cell to be somewhere in the row. All the cells between the `0` and the `B` are captured with `([^)]*)`. The regex ensures the color `B` exists by searching for the last non-zero, non-A color by using `(?!0|\4)(\d)`. Note that this never captures a `5` instead of `B` because `*` is greedy.

Then it simply captures up to the next `(stuff)` group using a lazy `.*?\1`, and then consumes the third `0` to be replaced with `B`, matching with the `B` on the third row of proposal 1.

The replacement is mostly just putting back everything that was consumed and replacing `0`s with `B`s, but there is a small pysearch slice for replacing the first half of the cells between the `0` and `B` on the middle row with the alternating `5, 0` pattern, which will get evaled into the correct replacement.

There are slight variations to this regex that are the same length, but we never found anything shorter than 201 bytes.

## task023:

In this task, light blue 2x2 squares and red 1x3 lines were placed to fill gray slots on the grid.

![task 23 preview](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28145366%2F16d0b086e80436e4048f5d89120f6152%2Ft023-preview.png?generation=1763018942991225&alt=media)

This task was tricky because there are many possible positions for the squares and lines but only one would work.

We found that the easiest way to solve this task was to use a recursive depth-first search.

Essentially, each call of the recursive function would find a suitable gray square, and check all shapes (1x3, 2x2, 3x1) at that location to see if they fit.
Then the algorithm would call itself with the shape retained, and eventually this would fill up the whole grid.

This is one example of a common theme in golfed solutions: **the most efficient example is rarely the shortest**.
There were no backtracking checks for 1x1 gray squares which would obviously mean no red or light blue shape could fit there, since that check
would take up extra bytes.

We also had to optimize for compression, since our solution was so long. To get the gray square, we found that simply getting the topmost leftmost gray square
using two loops would work since all previous shapes would be filled in going top-down left-right as well, so the optimal solution would always be found.

The shapes are encoded as bytestrings, and unpacked using *x,y where y is the color of the shape and x is the list of positive offsets, encoded using modulo.

We used byte strings and modulo to get the position offsets to be as short as possible. Also, try except was a short way to cancel the recursion if the shape
did not fit into the gray slots, so we used that, along with a non-existent variable, to cancel the loop. Lastly, bitwise operators allowed us to reuse the number `4`
a few times which would align nicely with the huffman tree during compression.

```py
def p(i):
 for u,m in enumerate(i):
  for r,f in enumerate(m):
   for*e,m in b"lur",b"lpt",b"lpum":
    try:
     for t in e:i[u+t%4][r+t%3]&4or n
     for t in e:i[u+t%4][r+t%3]=m
     if p(i):return i
     for t in e:i[u+t%4][r+t%3]=5
    except:i
   if f&4:return
 return i
```

We originally had the above which got compressed with our genetic varname compressor to be 198b. However, one more byte could be saved by using bytestring bytes that aligned
more nicely with the existing code (in huffman tree). The exact bytes and offsets (see `%4` and `%5` below) were brute forced for ones that seemed good.

```py
def p(e):
 for n,t in enumerate(e):
  for s,r in enumerate(t):
   for*b,t in b"(in",b"(tp",b"(ite":
    try:
     for i in b:e[n+i%4][s+i%5]&4or y
     for i in b:e[n+i%4][s+i%5]=t
     if p(e):return e
     for i in b:e[n+i%4][s+i%5]=5
    except:e
   if r&4:return
 return e
```

Other teams were able to get their solution shorter using regex, which we did not end up using for this task.

![play jail ctf 2026](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28145366%2F8ee82b71645b5352b210458ab9fac8f4%2FUntitled.png?generation=1763019431474135&alt=media)