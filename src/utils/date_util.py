import warnings
from .logger import logger
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

def convert_to_date(date_input, format=None):
    '''判断日期类型, 如果为字符串类型将其转为日期类型'''
    if isinstance(date_input, str):
        new_date = str_to_date(date_input, format)
    elif isinstance(date_input, datetime):
        new_date = date_input
    else:
        raise ValueError('无效的输入类型, 输入应该是字符串或日期时间对象.')
    return new_date

def str_to_date(str, format):
    '''字符串转日期'''
    return datetime.strptime(str, format)

def date_to_str(date, format):
    '''日期转字符串'''
    return datetime.strftime(date, format)

def get_first_day_of_month(date_input, format=None):
    '''获取月初日期
    参数：
        - date_input: 时间字符串或者日期类型
        - format: 时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - first_day: 月初日期
    '''
    date = convert_to_date(date_input, format)
    # 获取月初日期
    first_day = date.replace(day=1)
    return first_day

def get_first_day_of_quarter(date_input, format=None):
    '''获取本季度开始日期
    参数：
        - date_input: 时间字符串或者日期类型
        - format: 时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - first_day_of_quarter: 季度开始日期
    '''
    date = convert_to_date(date_input, format)
    # 获取所属季度的第一个月份
    quarter_start_month = (date.month - 1) // 3 * 3 + 1
    # 获取所属季度的第一天
    first_day_of_quarter = datetime(date.year, quarter_start_month, 1)
    return first_day_of_quarter


def get_first_day_of_year(date_input, format=None):
    '''获取本年度开始日期
    参数：
        - date_input: 时间字符串或者日期类型
        - format: 时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - first_day_of_year: 年度开始日期
    '''
    date = convert_to_date(date_input, format)
    # 获取年份
    year = date.year
    # 获取年度第一天
    first_day_of_year = datetime(year, 1, 1)
    return first_day_of_year


def get_quarter(date_input, format=None):
    '''获取日期所属季度
    参数：
        - date_input: 时间字符串或者日期类型
        - format: 时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - quarter: 年度开始日期
    '''
    date = convert_to_date(date_input, format)
    # 获取月份
    month = date.month
    quarter = (month - 1) // 3 + 1
    return quarter


def get_days_diff(date_input1, date_input2, format1=None, format2=None):
    '''获取给定两个日期的相差的天数,date_input2应比date_input1日期大
    参数：
        - date_input1: 时间字符串或者日期类型
        - date_input2: 时间字符串或者日期类型
        - format1: date_input1时间字符串的格式,例如'%Y-%m-%d'
        - format2: date_input2时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - days_diff: 相差的天数
    '''
    date1 = convert_to_date(date_input1, format1)
    date2 = convert_to_date(date_input2, format2)
    # 计算相差天数
    delta = date2 - date1
    days_diff = delta.days
    return days_diff

def get_months_diff(date_input1, date_input2, format1=None, format2=None):
    '''获取给定两个日期的相差的月份数,date_input2应比date_input1日期大
    参数：
        - date_input1: 时间字符串或者日期类型
        - date_input2: 时间字符串或者日期类型
        - format1: date_input1时间字符串的格式,例如'%Y-%m-%d'
        - format2: date_input2时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - months_diff: 相差的天数
    '''
    date1 = convert_to_date(date_input1, format1)
    date2 = convert_to_date(date_input2, format2)
    # 计算相差月份数
    # months_diff = (date2.year - date1.year) * 12 + date2.month - date1.month
    months_diff2 = (date2 - date1).days / 30
    return months_diff2

def get_years_diff(date_input1, date_input2, format1=None, format2=None):
    '''获取给定两个日期的相差的年数,date_input2应比date_input1日期大
    参数：
        - date_input1: 时间字符串或者日期类型
        - date_input2: 时间字符串或者日期类型
        - format1: date_input1时间字符串的格式,例如'%Y-%m-%d'
        - format2: date_input2时间字符串的格式,例如'%Y-%m-%d'

    返回：
        - months_diff: 相差的天数
    '''
    date1 = convert_to_date(date_input1, format1)
    date2 = convert_to_date(date_input2, format2)
    # 计算相差年数
    # years_diff = date2.year - date1.year
    years_diff2 = (date2 - date1).days / 365
    return years_diff2


def get_date_diff_of_day(date_input, days, format=None):
    '''获取给定日期相隔天数后的日期
    参数：
        - date_input: 时间字符串或者日期类型
        - days: int类型的天数

    返回：
        - new_date: 相隔天数的日期
    '''
    date = convert_to_date(date_input, format)
    # 计算相隔天数后的日期
    new_date = date + relativedelta(days=days)
    return new_date

def get_date_diff_of_month(date_input, months, format=None):
    '''获取给定日期相隔月份数后的日期
    参数：
        - date_input: 时间字符串或者日期类型
        - months: int类型的月份数

    返回：
        - new_date: 相隔月份数的日期
    '''
    date = convert_to_date(date_input, format)
    # 计算相隔月份数后的日期
    new_date = date + relativedelta(months=months)
    # new_date = get_date_diff_of_day(date, months*30)
    return new_date

def get_date_diff_of_year(date_input, years, format=None):
    '''获取给定日期相隔年数后的日期
    参数：
        - date_input: 时间字符串或者日期类型
        - years: int类型的年数

    返回：
        - new_date: 相隔年数的日期
    '''
    date = convert_to_date(date_input, format)
    # 计算相隔月份数后的日期
    new_date = date + relativedelta(years=years)
    # new_date = get_date_diff_of_day(date, years*365)
    return new_date

