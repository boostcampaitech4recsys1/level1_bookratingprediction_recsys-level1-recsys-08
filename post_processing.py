import time, os
import pandas as pd
from datetime import datetime

submission = pd.read_csv('./submit/' + '00221031_110112_FM.csv')

submission.loc[submission['rating'] > 10, 'rating'] = 10
submission.loc[submission['rating'] < 1, 'rating'] = 1

now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')
os.makedirs('submit', exist_ok=True)
submit_file_path = 'submit/post_{}.csv'.format(save_time)
submission.to_csv(submit_file_path, index=False)
print(f"Submit File Saved: {submit_file_path}")