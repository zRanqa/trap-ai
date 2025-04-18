

email_message = "helpThis Email is for you, check the attached doc for a SALES report, on the sales!! in the pastreport help"

key_words = [
    "sales",
    "report",
    "help"
]

email_words = email_message.replace(",","").split(" ")
email_contains = []

for i in range(len(email_message)):
    for j in range(len(key_words)):
        word_matches = True
        for k in range(len(key_words[j])):
            if i + k < len(email_message):
                if email_message[i + k].lower() != key_words[j][k]:
                    word_matches = False
            else:
                word_matches = False
        if word_matches:
            email_contains.append(key_words[j])
            print(f"word found at index {i}")


print("This email message contains:")
for word in email_contains:
    print(f"\t{word}")
