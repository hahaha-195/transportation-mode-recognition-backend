# from meteostat import Daily
# from datetime import datetime
#
# # 北京气象站（54511）
# start = datetime(2007, 1, 1)
# end   = datetime(2012, 12, 31)
#
# data = Daily('54511', start, end)
# data = data.fetch()
#
# # 保存为 CSV
# data.to_csv('beijing_weather_2007_2012.csv')
#
# print(data.head())

from meteostat import Hourly
from datetime import datetime

start = datetime(2007, 1, 1)
end   = datetime(2012, 12, 31)

data = Hourly('54511', start, end)
data = data.fetch()

print(data.head())

data.to_csv('beijing_weather_hourly_2007_2012.csv')
