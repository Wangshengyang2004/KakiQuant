import datetime
def today_is(date):
    """Returns True if today is the given date, False otherwise."""
    return date == datetime.date.today()

def today(tushare_format=False):
    if tushare_format:
        return datetime.date.today().strftime("%Y%m%d")
    return datetime.date.today()

def current_timestamp():
    return datetime.time.microsecond()


if __name__ == "__main__":
    print(today(), current_timestamp())
