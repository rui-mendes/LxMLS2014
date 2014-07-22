__author__ = 'ruimendes'


def greetOld(hour):
    if hour < 12:
        print 'Good morning!'
    elif hour >= 12 and hour < 20:
        print 'Good afternoon!'
    else:
        print 'Good evening!'

"""
Note that the previous code allows the hour to be less than 0 or more than 24. Change the code in order to
indicate that the hour given as input is invalid. Your output should be something like:
greet(50)
Invalid hour: it should be between 0 and 24.
greet(-5)
Invalid hour: it should be between 0 and 24.
"""


def greet(hour):
    if hour < 0 or hour > 24:
        print 'Invalid hour: it should be between 0 and 24.'
    elif hour < 12:
        print 'Good morning!'
    elif hour >= 12 and hour < 20:
        print 'Good afternoon!'
    else:
        import ipdb
        ipdb.set_trace()
        print 'Good evening!'

greet(20)
greet(-5)