def parseLines(data):
    res = []
    for line in data:
        if line.startswith("[INFO] Predicted 0 Input: "):
            seq = line.split(":")[1]
            res.append(seq.strip())
    return res


file_name_1 = "/Users/jianxiongcai/Downloads/log_base_model"
file_name_2 = "/Users/jianxiongcai/Downloads/log_defense_model"

with open(file_name_1, "r") as f:
    data_1 = f.readlines()
    error_1 = parseLines(data_1)

with open(file_name_2, "r") as f:
    data_2 = f.readlines()
    error_2 = parseLines(data_2)

print()
print("======================================================================================================")
print("                          Common Errors:                          ")
print("======================================================================================================")
for seq in error_1:
    if seq in error_2:
        print(seq)

print()
print("======================================================================================================")
print("                          Error Corrected by Defense Model:                          ")
print("======================================================================================================")
for seq in error_1:
    if seq not in error_2:
        print(seq)




