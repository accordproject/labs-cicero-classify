US_States = ['AL',
 'AK',
 'AZ',
 'AR',
 'CA',
 'CO',
 'CT',
 'DE',
 'FL',
 'GA',
 'HI',
 'ID',
 'IL',
 'IN',
 'IA',
 'KS',
 'KY',
 'LA',
 'ME',
 'MD',
 'MA',
 'MI',
 'MN',
 'MS',
 'MO',
 'MT',
 'NE',
 'NV',
 'NH',
 'NJ',
 'NM',
 'NY',
 'NC',
 'ND',
 'OH',
 'OK',
 'OR',
 'PA',
 'RI',
 'SC',
 'SD',
 'TN',
 'TX',
 'UT',
 'VT',
 'VA',
 'WA',
 'WV',
 'WI',
 'WY']
def is_US_States(text):
    return text in US_States

def label_US_States(row):
    if is_US_States(row.Word):
        if row.Tag == "O":
            row.Tag = "|".join(["B-geo", 'Location', 'Party', "US_State"])
        else:
            row.Tag = row.Tag + "|" + "|".join(['Location', 'Party', "US_State"])