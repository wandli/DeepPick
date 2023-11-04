from pprint import pprint


def load_data(mode):
    special_users = {}
    with open('./before_clean/' + mode + '_text.csv', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        lines = [line.split(',') for line in lines]
        del lines[0]
        for line in lines:
            if line[0] in special_users.keys():
                special_users[str(int(float(line[0])))] += (' ' + line[1])
            else:
                special_users[str(int(float(line[0])))] = line[1]

    ordinary_users = {}
    with open('./before_clean/' + mode + 'ord_text.csv', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        lines = [line.split(',') for line in lines]
        del lines[0]
        for line in lines:
            if line[0] in ordinary_users.keys():
                ordinary_users[str(int(float(line[0])))] += (' ' + line[1])
            else:
                ordinary_users[str(int(float(line[0])))] = line[1]

    return special_users, ordinary_users


def save_data(mode, special_users, ordinary_users):
    with open('./after_clean/' + mode + '_text.csv', 'a', encoding='utf-8') as f:
        for key, value in special_users.items():
            if len(value.split(' ')) >= 5:
                f.write(key + ',' + value + '\n')

    with open('./after_clean/' + mode + 'ord_text.csv', 'a', encoding='utf-8') as f:
        for key, value in ordinary_users.items():
            if len(value.split(' ')) >= 5:
                f.write(key + ',' + value + '\n')


def main():
    # super parameters
    mode = 'shell'
    special_users, ordinary_users = load_data(mode)
    save_data(mode, special_users, ordinary_users)


if __name__ == '__main__':
    main()

