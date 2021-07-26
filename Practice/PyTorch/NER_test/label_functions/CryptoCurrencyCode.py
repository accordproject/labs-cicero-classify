CryptoCurrencyCodes = [
    'ADA',
    'BCH',
    'BTC',
    'DASH',
    'EOS',
    'ETC',
    'ETH',
    'LTC',
    'NEO',
    'XLM',
    'XMR',
    'XRP',
    'ZEC',
]
def is_CryptoCurrencyCode(text):
    return text.upper() in CryptoCurrencyCodes