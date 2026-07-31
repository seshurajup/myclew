"""Author the 21 legacy videos (01-learn-python, 03-learn-transformer) up to standard.

These predate the 81-video curriculum: 5-7 narration segments each (~22-40s), several with
until_line / after_line values pointing past the end of their code.py. This script rewrites each
transcript to 9-10 segments (~60s of real narration, per README "length = content, never silence"),
fixes the out-of-range pins, and sets the promise-style hook.

Run once:  python tools/author_legacy.py
Then:      python tools/apply_retention_rules.py tools/hooks.tsv   (adds the next-video tails)
"""
import json
from pathlib import Path

YT = Path(__file__).resolve().parent.parent

# {video_dir: [(until_line, text), ...]} — pins verified against each code.py line count
T = {
"01-learn-python/001-fibonacci": [
 (1, "Fibonacci is the sequence where every number is the sum of the two before it, and we'll build it in eight lines."),
 (1, "The signature says it all — take an integer n, hand back a list of integers."),
 (2, "A docstring on line two documents what the function does for whoever reads it next."),
 (3, "We seed the list with zero and one, because the sequence needs two numbers before it can grow."),
 (4, "Then we loop while the list is still shorter than n — that's our stopping condition."),
 (5, "Each pass appends the sum of the last two entries. Minus one is the final item, minus two the one before it."),
 (5, "Negative indexing reads from the end backwards, so this stays readable however long the list gets."),
 (6, "The final slice quietly handles the edge case where n is smaller than two."),
 (8, "And printing the first ten shows that classic curve — small, then suddenly not."),
],
"01-learn-python/002-list-comprehensions": [
 (1, "A list comprehension builds a whole list in one readable line, with no append loop required."),
 (1, "This one squares every number from zero to nine — the expression comes first, then the for."),
 (2, "Add an if on the end and it becomes a filter, so evens keeps only the multiples of two."),
 (3, "You can even nest two loops in a single comprehension, left to right, exactly like nested for statements."),
 (3, "This builds every x y pair where the two differ, skipping the diagonal."),
 (3, "Read it out loud and it says what it does — give me x times x, for x in range ten."),
 (5, "Now let's print all three and see exactly what we built."),
 (7, "Squares, evens and pairs — three data structures, three lines."),
 (7, "And not a single manual append anywhere."),
],
"01-learn-python/003-decorators": [
 (1, "A decorator is just a function that wraps another function and changes what it does."),
 (1, "Shout takes a function called func and returns a brand new function built around it."),
 (2, "The inner wrapper accepts anything at all — star args and double-star kwargs catch every argument."),
 (3, "Inside, it calls the original function and keeps the result."),
 (4, "Then it upper-cases that result and adds an exclamation mark before handing it back."),
 (5, "Line five returns the wrapper itself, not a call to it — that distinction matters."),
 (8, "Now watch the at-sign syntax sitting right above greet."),
 (10, "At-shout quietly replaces greet with shout of greet — same name, new behaviour."),
 (13, "So calling greet with world doesn't politely say hello, it shouts it."),
 (13, "Same function, zero changes to its body."),
],
"01-learn-python/004-generators": [
 (1, "A generator produces values one at a time instead of building a whole list up front."),
 (1, "Countdown looks like a normal function, right up until you spot the yield."),
 (3, "While n is greater than zero we yield n — and yield is the key word here."),
 (3, "Yield hands a value back and pauses the function, keeping every local variable alive."),
 (4, "The next time you ask for a value it resumes on the very next line and decrements n."),
 (5, "Once the loop finally ends, we sneak in one last yield, the string liftoff."),
 (8, "A for loop just keeps pulling values until the generator is exhausted."),
 (9, "Three, two, one, liftoff — each one arriving only when it's asked for."),
 (9, "No list is ever built in memory, which is why this scales to millions of items."),
],
"01-learn-python/005-context-managers": [
 (1, "A context manager guarantees that cleanup happens, no matter how your code exits."),
 (1, "Any class with two special methods can be one — enter and exit."),
 (2, "Enter runs when the with block starts, and whatever it returns is what you bind with as."),
 (4, "Here it prints starting and hands back the object itself."),
 (6, "Exit runs when the block finishes, and star exc catches any exception details Python passes in."),
 (8, "It prints done, then returns False, meaning don't suppress errors — let them propagate."),
 (8, "If exit returned True instead, it would swallow the exception entirely, which is rarely what you want."),
 (11, "Now the with statement itself. Timer parentheses creates the object and enters it."),
 (11, "This is exactly how open works when you read a file with a with statement."),
 (12, "Inside the block we do the actual work."),
 (12, "Starting, then working, then done — and that done is guaranteed even if the body raises."),
],
"01-learn-python/006-dataclasses": [
 (1, "A dataclass writes the boring parts of a class for you."),
 (1, "We import the decorator from the dataclasses module in the standard library."),
 (4, "At-dataclass sits above the class, and that single line is doing all the work."),
 (7, "You just declare the fields with type annotations — x is an int, y is an int."),
 (7, "From those two lines Python generates the constructor, plus equality and a readable repr."),
 (9, "You can still add your own methods; a dataclass is a completely normal class underneath."),
 (10, "Dist computes the distance from the origin, squaring both fields and taking the square root."),
 (13, "Now we build a Point with three and four, positionally, thanks to the generated init."),
 (14, "Printing it gives Point x equals three, y equals four — not an ugly memory address."),
 (15, "And dist returns five point zero, the three four five triangle, from a class we barely wrote."),
],
"01-learn-python/007-error-handling": [
 (1, "Try, except and finally let you handle failure without your program falling over."),
 (1, "Safe div takes two numbers and attempts the division inside a try block."),
 (3, "Everything in the try is protected — if it succeeds, the return happens and we move on."),
 (4, "Except catches one specific error, ZeroDivisionError, not every error in existence."),
 (4, "Catching narrowly matters; a bare except would swallow your typos and bugs too."),
 (5, "When the divide by zero does happen, we return None instead of crashing."),
 (7, "Finally always runs — success or failure, return or exception, no way around it."),
 (7, "That's what makes it the right place for cleanup, like closing a file."),
 (10, "Ten over two gives five, and checked prints alongside it."),
 (11, "Ten over zero returns None instead of blowing up, and checked still prints."),
],
"01-learn-python/008-itertools": [
 (1, "Itertools is a standard library module full of tools for looping smarter."),
 (1, "We're pulling in two of them, chain and combinations."),
 (1, "Everything in it returns an iterator, so nothing is computed until you ask for it."),
 (3, "Chain glues iterables together and walks them as if they were one sequence."),
 (4, "It doesn't copy anything — it just steps through each source in turn, lazily."),
 (4, "Notice the lists hold different types, numbers and strings, and chain doesn't care."),
 (4, "That laziness is the point — you can chain over a file of a million lines without loading it."),
 (6, "Combinations gives every way of picking k items, ignoring order."),
 (7, "With three numbers taken two at a time, that's one-two, one-three and two-three."),
 (7, "Order never repeats, so two-one is not counted separately."),
 (7, "Both return lazy iterators, so we wrap them in list to actually see the values."),
 (7, "Combinations is how you generate every pair from a list without a nested loop."),
],
"01-learn-python/009-fstrings": [
 (1, "F-strings do far more than drop a variable into some text."),
 (2, "We start with a float and a short string to play with."),
 (4, "Put an f before the quote, and anything in curly braces gets evaluated."),
 (4, "Colon dot two f is a format spec — round this float to two decimal places."),
 (4, "So pi prints as three point one four, not the full messy value."),
 (4, "Format specs handle padding and thousands separators too, all inside the braces."),
 (5, "Bang r calls repr instead of str, which keeps the quotes visible around the string."),
 (5, "And you can run real expressions inside the braces, like len of name."),
 (8, "The equals sign suffix is the debugging trick — it prints the expression and its value."),
 (8, "x equals five, written once, shown as both the name and the result."),
 (8, "That saves you writing the variable name twice every time you debug."),
],
"01-learn-python/010-walrus": [
 (1, "The walrus operator assigns a value and returns it, in the same breath."),
 (1, "Before it existed, you had to assign on one line and test on the next."),
 (3, "We start with six numbers, an empty result list, and an index at zero."),
 (5, "Now the while condition — and there's a lot packed into this one line."),
 (5, "Colon equals assigns to n and hands that same value straight to the comparison."),
 (5, "The conditional expression guards the end of the list, giving None once we run out."),
 (5, "So n is assigned and tested at once, with no duplicate line above the loop."),
 (5, "The parentheses around the walrus are required here, and usually a good idea anyway."),
 (7, "Inside, we keep only the even numbers."),
 (8, "And we bump the index so the loop actually moves forward."),
 (10, "Two, four and six come out — the walrus saved us a line and a repeated read."),
],
"01-learn-python/011-type-hints": [
 (1, "Type hints tell you and your tools exactly what a function expects."),
 (1, "Name is a string, times is an integer defaulting to one, and it returns a string."),
 (1, "Python doesn't enforce any of this at runtime — it's documentation your editor can check."),
 (1, "They cost you nothing at runtime and save you real time when the code grows."),
 (2, "The body repeats the greeting and strips the trailing space."),
 (4, "Variables take annotations too. Scores is a dict mapping string to int."),
 (5, "Top is a list of strings, built by a comprehension over the items."),
 (5, "Anyone reading line five now knows what's inside without running a thing."),
 (7, "Calling greet with just a name uses the default and greets once."),
 (9, "Pass three and you get it three times, and top gives us the high scorers."),
 (9, "Run mypy over this file and it checks every one of those promises for you."),
],
"01-learn-python/012-enumerate-zip": [
 (1, "Enumerate and zip are the clean way to loop when you need more than one thing."),
 (2, "We've got three names and three matching ages in parallel lists."),
 (4, "Enumerate hands you the index and the value together on every pass."),
 (4, "And start equals one means you count from one, not zero — ideal for numbered output."),
 (5, "So you never write range len and index back into the list by hand."),
 (7, "Zip walks two sequences side by side, pairing them position by position."),
 (8, "Each pass unpacks straight into name and age, ready to use."),
 (8, "Zip stops at the shorter of the two, so it never runs off the end."),
 (10, "And wrapping zip in dict turns two parallel lists into one dictionary, instantly."),
 (11, "Names to ages, in a single readable line."),
],
"01-learn-python/013-counter": [
 (1, "Counter tallies anything countable, and it does it in one line."),
 (1, "It lives in collections, alongside the other specialised containers."),
 (1, "It's the one-line answer to counting words, votes, or anything else that repeats."),
 (3, "We split a sentence into words, giving us a list with repeats."),
 (4, "Passing that list to Counter counts every element as it goes."),
 (6, "Printing it shows each word with its tally, ordered most common first."),
 (7, "You index it like a dictionary, because a Counter is a dictionary underneath."),
 (7, "Red appears three times, so counts of red gives three."),
 (8, "Most common one gives you the single biggest entry, as a list of pairs."),
 (8, "Pass a bigger number and you get that many of the top entries back."),
 (10, "Update adds more items to the existing tally rather than replacing it."),
 (11, "Blue was two, we added two more, and now it's four."),
],
"01-learn-python/014-defaultdict": [
 (1, "Defaultdict removes the most annoying line you write when grouping data."),
 (1, "It comes from collections, the same module as Counter."),
 (3, "We've got pairs of a category and an item, with fruit appearing twice."),
 (4, "Defaultdict list means any missing key starts life as an empty list."),
 (4, "That's the whole trick — you hand it a factory, and it calls that factory on a miss."),
 (7, "Now the loop just appends, with no checking whether the key exists yet."),
 (7, "With a plain dict, line seven would raise a KeyError the very first time round."),
 (9, "Fruit collected apple and pear, exactly as they arrived."),
 (10, "Veg has its own list, created the moment it was first touched."),
 (11, "And wrapping it in dict gives you back an ordinary dictionary when you're done."),
],
"01-learn-python/015-lambda-map-filter": [
 (1, "Lambda, map and filter let you transform a list without writing a loop."),
 (1, "We start with six numbers to work on."),
 (3, "A lambda is a function with no name — arguments, colon, and a single expression."),
 (3, "Map applies that lambda to every element, one at a time."),
 (3, "There's no return statement, because the expression itself is the return value."),
 (4, "Filter keeps only the elements where the lambda returns True."),
 (4, "Here that's the even numbers, since x mod two equals zero."),
 (4, "Filter keeps the element itself, not the True or False the lambda produced."),
 (6, "Both are lazy, so list forces them to actually produce their values."),
 (7, "Squares gives us the six squares, and evens gives us two, four and six."),
 (9, "And they compose — map times ten over evens, summed in a single expression."),
 (10, "Twenty plus forty plus sixty is a hundred and twenty."),
],
"01-learn-python/016-args-kwargs": [
 (1, "Star args and double-star kwargs let a function accept any number of arguments."),
 (1, "Star args collects every positional argument into a tuple called args."),
 (2, "So total just sums that tuple, however many numbers you pass in."),
 (2, "Even calling it with no arguments works, because sum of an empty tuple is zero."),
 (4, "Double-star kwargs does the same for keyword arguments, collecting them into a dictionary."),
 (5, "We join the key value pairs into one readable string."),
 (5, "The names args and kwargs are only convention — the stars do the actual work."),
 (7, "Four numbers in, and total adds them all to ten."),
 (8, "Profile takes name and age as keywords and formats them back out."),
 (10, "The star works in the other direction too, when you're calling rather than defining."),
 (11, "Star nums unpacks the list into three separate arguments, giving sixty."),
 (11, "One star unpacks a sequence, and two stars unpack a dictionary into keywords."),
],
"01-learn-python/017-unpacking": [
 (1, "Unpacking pulls values out of a sequence and into variables, in one line."),
 (1, "First takes the head, last takes the tail, and star middle absorbs everything between."),
 (1, "Star middle always becomes a list, even when it catches nothing at all."),
 (1, "You can only use one star in a target list, since two would be ambiguous."),
 (2, "So first and last are one and five."),
 (3, "And middle is two, three, four, collected automatically."),
 (5, "Unpacking also gives you the cleanest swap in any language."),
 (6, "a, b equals b, a. The right side is built first, so no temporary variable is needed."),
 (7, "Two and one, swapped without a third name."),
 (10, "Double star unpacks dictionaries, and later keys win over earlier ones."),
 (10, "That makes it the neatest way to override defaults without mutating the original."),
 (11, "So colour becomes blue while size keeps its default of M."),
],
"01-learn-python/018-sets": [
 (1, "A set is an unordered collection of unique values, with real maths built in."),
 (1, "Duplicates simply cannot exist in a set — adding one again changes nothing."),
 (2, "We define two sets that share the numbers three and four."),
 (4, "Ampersand is intersection — only the elements found in both."),
 (4, "That gives us three and four, the overlap."),
 (5, "Pipe is union, every element from either side, with duplicates collapsed."),
 (5, "Union is how you merge two collections and forget about overlap entirely."),
 (6, "Minus is difference, what's in a but not in b."),
 (6, "Which leaves one and two."),
 (9, "Passing a list to set is the fastest way to remove duplicates."),
 (9, "Six items become three, though the order is not guaranteed."),
 (10, "And membership tests are near instant, because sets are hash based, not scanned."),
 (10, "Checking membership in a list scans it; in a set it's a single hash lookup."),
],
"01-learn-python/019-sorting-key": [
 (1, "Sorted can order anything, as long as you tell it what to sort by."),
 (1, "We start with four words of different lengths."),
 (1, "Sorted always returns a new list, leaving the original untouched."),
 (3, "Key equals len means sort by the length of each word, not alphabetically."),
 (3, "Notice we pass len itself, not len parentheses — the function, not a call to it."),
 (4, "So fig, kiwi, apple, banana. Shortest to longest."),
 (7, "For tuples, a lambda picks which part to sort on."),
 (7, "p bracket one grabs the age, and reverse equals True flips it to highest first."),
 (7, "Any function works as a key, so you can sort by anything you can compute."),
 (8, "Cat at forty, amy at thirty, ben at twenty-five."),
 (10, "Max takes the very same key argument, and so do most Python functions."),
 (10, "So the oldest person falls out with no sorting at all."),
],
"01-learn-python/020-pathlib": [
 (1, "Pathlib replaces string juggling with real path objects."),
 (1, "We import Path, the one class you'll use ninety percent of the time."),
 (1, "Every one of these paths is an object with methods, not a fragile string."),
 (3, "The forward slash is overloaded to join paths, and it's correct on every operating system."),
 (3, "No backslashes, no os dot path dot join, no separators to get wrong."),
 (3, "Path objects also compare, sort and work as dictionary keys."),
 (5, "Dot name gives the final component, report dot txt."),
 (6, "Dot suffix is just the extension, dot txt."),
 (7, "Dot stem is the name without it, report."),
 (8, "Dot parent walks up one level, to data."),
 (9, "And with suffix swaps the extension cleanly, giving report dot md."),
 (9, "And with suffix returns a new path, it never mutates the one you had."),
 (9, "Read text, write text and exists hang off the same object when you need them."),
],
"03-learn-transformer/001-attention": [
 (1, "Scaled dot product attention is the heart of the transformer, and it's four lines of PyTorch."),
 (1, "We import torch, and that's the only dependency."),
 (3, "The function takes Q, K and V — query, key and value."),
 (4, "d_k is the size of the last dimension, the width of each vector."),
 (5, "Q at K transposed is a matrix of dot products, every query against every key."),
 (5, "That dot product is the similarity score, how much each token should attend to each other token."),
 (5, "We divide by the square root of d_k to stop those scores growing with dimension."),
 (6, "Without that scaling, softmax saturates and the gradients vanish."),
 (6, "Softmax turns each row of scores into weights that sum to one."),
 (7, "Then weights at V takes a weighted average of the values."),
 (12, "Six tokens, eight dimensions each, and the output comes back the same shape."),
],
}

# out-of-range pins in outputs.json, corrected to the real last line of code.py
OUTPUT_FIX = {
    "01-learn-python/002-list-comprehensions": 7,
    "01-learn-python/015-lambda-map-filter": 10,
    "01-learn-python/016-args-kwargs": 11,
    "03-learn-transformer/001-attention": 12,
}


def main():
    for key, segs in T.items():
        d = YT / key
        nlines = len((d / "code.py").read_text().rstrip("\n").split("\n"))
        bad = [ul for ul, _ in segs if not 1 <= ul <= nlines]
        assert not bad, f"{key}: until_line {bad} outside 1..{nlines}"
        assert max(ul for ul, _ in segs) >= nlines - 2, f"{key}: code won't finish typing"
        (d / "transcript.json").write_text(
            json.dumps([{"text": t, "until_line": ul} for ul, t in segs],
                       indent=2, ensure_ascii=False) + "\n")

        # every legacy video was 22-40s; give the builder room for the real narration
        spec_p = d / "spec.json"
        spec = json.loads(spec_p.read_text())
        spec["target_seconds"], spec["max_seconds"] = 90, 120
        spec_p.write_text(json.dumps(spec, indent=2) + "\n")

        if key in OUTPUT_FIX:
            outs_p = d / "outputs.json"
            outs = json.loads(outs_p.read_text())
            for o in outs:
                if o.get("after_line") is not None:
                    o["after_line"] = min(o["after_line"], OUTPUT_FIX[key])
            outs_p.write_text(json.dumps(outs, indent=2, ensure_ascii=False) + "\n")

        words = sum(len(t.split()) for _, t in segs)
        print(f"{key:44} {len(segs):2} segs  ~{words/2.6:.0f}s")
    print(f"\n{len(T)} legacy videos rewritten")


if __name__ == "__main__":
    main()
