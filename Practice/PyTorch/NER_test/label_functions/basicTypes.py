def is_Integer(text):
    try:
        int(text.replace(",", ""))
        return True
    except:
        return False
    
def is_Float(text):
    if "." in text:
        try:
            float(text)
            return True
        except:
            return False
    return False



