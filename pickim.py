import random

random.seed(3)


im_names = []
scores = []
count = 0
with open("error_image_low_1.txt","r") as f:
    for line in f.readlines():
        count += 1
        if count<=50:
            line = line.strip('\n')
            name, score = line.split()
            im_names.append(name)
            scores.append(score)
            print(count, line) 


sorted_id=sorted(range(len(scores)), key=lambda k:scores[k], reverse=True)
'''
top_names = [im_names[i] for i in sorted_id][0:round(len(scores)/2)]
print(len(top_names), top_names)
high_text = open("high_2.txt","w")
for elem in top_names:
    high_text.write(elem[:-4]+"\n")
high_text.close()

bottom_names = [im_names[i] for i in sorted_id][round(len(scores)/2):]
print(len(bottom_names), bottom_names)
low_text = open("low_2.txt","w")
for elem in bottom_names:
    low_text.write(elem[:-4]+"\n")
low_text.close()
'''
random_names = random.sample(im_names,round(len(scores)/2))
print(len(random_names), random_names)
rand_text = open("rand_2.txt","w")
for elem in random_names:
    rand_text.write(elem[:-4]+"\n")
rand_text.close()
'''
all_text = open("all_1.txt","w")
for elem in im_names:
    all_text.write(elem[:-4]+"\n")
all_text.close()
'''

