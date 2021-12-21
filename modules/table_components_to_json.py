import pandas as pd


def nest(d: dict) -> dict:
    result = {}
    for keys, value in d.items():
        target = result

        if isinstance(keys, tuple):
            first_key, second_key = keys

            if second_key:
                target = target.setdefault(first_key, {})
                target[second_key] = value
            else:
                target[first_key] = value
        else:
            first_key = keys
            target[first_key] = value
        
    return result


def df_to_nested_dict(df: pd.DataFrame) -> dict:
    dict_list = df.to_dict(orient='records')
    result = []
    for i, d in enumerate(dict_list):
        current_result = {
            'row_number': i,
        }
        current_result.update(nest(d))
        result.append(current_result)
    return result


def combine_body_head(body, head):
    if all([isinstance(item, tuple) for item in head]):
        body.columns = pd.MultiIndex.from_tuples(head)
    elif all([isinstance(item, str) for item in head]):
        body.columns = head
    else:
        pass
    return body


def process_table(body, head):
    df = combine_body_head(body, head)
    result = df_to_nested_dict(df)
    return result