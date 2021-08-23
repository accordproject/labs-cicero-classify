months = ['January',
 'February',
 'March',
 'April',
 'May',
 'June',
 'July',
 'August',
 'September',
 'October',
 'November',
 'December']

months_short_form = ['Jan.',
 'Feb.',
 'Mar.',
 'Apr.',
 'May',
 'Jun.',
 'Jul.',
 'Aug.',
 'Sep.',
 'Oct.',
 'Nov.',
 'Dec.']

all_form_month = months + months_short_form

def isMonth(text):
    return text.capitalize() in all_form_month