import glob
import os

# with open('record', 'w') as f:
#     for name in glob.glob('/home/eegroup/ee50523/b08901019/dlcv_proj_workspace/student_data/videos/*'):
#         f.write(name.split('/')[-1][:-4])
#         f.write('\n')

# with open('record_all', 'r') as a:
#     with open('record', 'r') as b:
#         with open('loss_name', 'w') as f:
#             all_name = a.readlines()
#             part_name = b.readlines()
#             print( len(all_name), len(part_name) )
#             for idx in range( len(all_name) ):
#                 for find in range( len(part_name) ):
#                     if(all_name[idx] == part_name[find]):
#                         break
#                     if(find == len(part_name)-1):
#                         f.write(all_name[idx])
                        
#             print( len(all_name)-len(part_name) )
# with open('loss_name', 'r') as r:
#     c = r.readlines()
#     print(len(c))

total = 0

for whole_video_path in glob.glob('./videos/*'):

    name = whole_video_path.split('/')[-1]

    if os.path.exists(f'./train/seg/{name[:-4]}_seg.csv'):
        with open(f'./train/seg/{name[:-4]}_seg.csv') as dur_file:
            infos = dur_file.readlines()
            total += len(infos)

    if os.path.exists(f'./test/seg/{name[:-4]}_seg.csv'):
        with open(f'./test/seg/{name[:-4]}_seg.csv') as dur_file:
            infos = dur_file.readlines()
            total += len(infos)

print(total)