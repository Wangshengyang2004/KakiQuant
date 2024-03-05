import datetime
def today_is(date):
    """Returns True if today is the given date, False otherwise."""
    return date == datetime.date.today()

def today(tushare_format=False):
    if tushare_format:
        return datetime.date.today().strftime("%Y%m%d")
    return datetime.date.today()

def date_to_datetime(date: str):
    """Converts a date to a datetime object with time set to 00:00:00."""
    return datetime.datetime(date.year, date.month, date.day)
