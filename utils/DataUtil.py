import jsonpickle

def main():
    dataDir = "../UMR Data"

    doc_aligned_with_joined_sentence = open(dataDir + "/doc_aligned_with_joined_sentence.json", 'w')

    with open(dataDir + "/doc_level_blanks_aligned.json", 'r') as f:
        data = jsonpickle.decode(f.read())

        for i in range(len(data)):
            data[i]['joined_sentence'] = ' '.join(data[i]['sentence'].values())
            data[i]['mapping'] = {}

            for item in data[i]['alignment']:
                key = item.split(":")[0]
                item = item.split(":")[1].strip().split('-')

                data[i]['mapping'][key] = [int(item[0]), int(item[-1])]

        doc_aligned_with_joined_sentence.write(jsonpickle.encode(data))

    doc_aligned_with_joined_sentence.close()

if __name__ == "__main__":
    main()