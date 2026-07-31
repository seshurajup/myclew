import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

log = logging.getLogger("app")

log.debug("detailed info, hidden by default")
log.info("the app started")
log.warning("low disk space")
log.error("something failed")

# logging captures context print never could
def process(item):
    log.info("processing %s", item)
    return item.upper()

print(process("data"))
