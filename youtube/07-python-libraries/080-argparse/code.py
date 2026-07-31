import argparse

parser = argparse.ArgumentParser(description="Greet someone")
parser.add_argument("name", help="who to greet")
parser.add_argument("--times", type=int, default=1, help="repeat count")
parser.add_argument("--shout", action="store_true", help="uppercase")

# simulate: python greet.py Alice --times 2 --shout
args = parser.parse_args(["Alice", "--times", "2", "--shout"])

msg = f"Hello, {args.name}!"
if args.shout:
    msg = msg.upper()
for _ in range(args.times):
    print(msg)
