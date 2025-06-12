# Utils class to handle some utility functions
import json
def ReadKeyMapValueFromFile(file_path):
    result = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                index_part, text_part = line.split(':', 1)
                text_part  = text_part[0:-2]
                key = int(index_part.strip())
                value = text_part.replace("'", "").strip()
                result[key] = value
    return result