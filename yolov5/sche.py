import schedule
import time
import Copy_detect3

schedule.every(10).seconds.do(Copy_detect3.main, opt = Copy_detect3.parse_opt)

i = 0
while True:
    schedule.run_pending()
    print(i)
    time.sleep(1)
    i = i + 1
