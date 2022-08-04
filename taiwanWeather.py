# %%
# %%
import pandas as pd
pd.set_option('display.max_rows', 200)
import pandasql as ps
import numpy as np
from dateutil.parser import parse
import math
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
import requests
import json
from geopy.distance import distance

# %%
class taiwanWeather:
    def get_weather_data (self, sta_no = 'C0A520', start_date = '20210816', end_date = '20210913', interpolation = False, type = 'hourly'):
        start_date = parse(start_date).strftime('%Y%m%d')
        end_date = parse(end_date).strftime('%Y%m%d')
        #URI = "http://mycolab.pp.nchu.edu.tw/CODIS_downloader/index.php?station_id={}&startdate={}&enddate={}".format(sta_no, start_date, end_date)
        URI = "http://mycolab.pp.nchu.edu.tw/historical_weather/index.php?station_id={}&startdate={}&enddate={}&type={}".format(sta_no, start_date, end_date, type)
        #print (URI)
        try:
            df = pd.read_csv(URI)
        except Exception as e:
            return pd.DataFrame()
        DATE_COL = '觀測時間(hour)' if type == 'hourly' else '觀測時間(day)'
        #debug
        #df.to_csv('./debug/{}_{}_{}_KeyError.csv'.format(sta_no, start_date, end_date))
        try:
            #Sometimes the observation data will contain data with all zeros instead of -999
            #so we have to convert them to -999 manually in order to catch them all when interpolating.
            df_time_bak = df[DATE_COL].copy()
            df[df['測站氣壓(hPa)']==0] = -999
            df[df['測站氣壓(hPa)']==0] = -999
            df[DATE_COL] = df_time_bak

            #Calculate wv and wu
            wv = df['風速(m/s)']
            wd_rad = df['風向(360degree)']*np.pi / 180
        except KeyError:
            return pd.DataFrame()
        #Post-processing  wind
        df['Wv'] = round(-wv*np.cos(wd_rad),2)
        df['Wu'] = round(-wv*np.sin(wd_rad),2)
        df['Wv'] = df['Wv'].apply(self.invalid_wu_wv_to_999)
        df['Wu'] = df['Wu'].apply(self.invalid_wu_wv_to_999)
        #Post-processing  date
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df['Month_normalized'] =  df[DATE_COL].dt.month
        df['Month_normalized'] = round(np.sin(df['Month_normalized']*2*np.pi/12),2)
        df['Hour_normalized'] =  df[DATE_COL].dt.hour
        df['Hour_normalized'] = round(np.sin(df['Hour_normalized']*2*np.pi/24),2)

        if interpolation:
            consecutive_false_threshold = 48
            if (self.count_consecutive_false(df['氣溫(℃)'].apply(self.is_valid)) >consecutive_false_threshold or
                self.count_consecutive_false(df['相對溼度(%)'].apply(self.is_valid)) >consecutive_false_threshold or
                self.count_consecutive_false(df['風速(m/s)'].apply(self.is_valid)) >consecutive_false_threshold or
                self.count_consecutive_false(df['降水量(mm)'].apply(self.is_valid)) >consecutive_false_threshold ):
                
                print ('Skipped, too many invalid data')
                return pd.DataFrame()
            else:
                try:
                    df['氣溫(℃)'] = self.auto_cubicspline_interpolation(df['氣溫(℃)'])
                    df['相對溼度(%)'] = self.auto_cubicspline_interpolation(df['相對溼度(%)'])
                    df["相對溼度(%)"] = df["相對溼度(%)"].apply(lambda x: min(x,100)) #set upper bound of RH
                    df['Wv'] = self.auto_cubicspline_interpolation(df['Wv'])
                    df['Wu'] = self.auto_cubicspline_interpolation(df['Wu'])
                    df['降水量(mm)'] = self.auto_cubicspline_interpolation(df['降水量(mm)'])
                except:
                    return pd.DataFrame()
        return df 

    def is_valid(self,s):
        try: 
            float(s)
            if (float(s)<-99 or float(s)==-9.8 or math.isnan(float(s))):
                return False
            return True
        except ValueError:
            return False

    def invalid_wu_wv_to_999(self,s):
        if s==float('nan') or s>=50 or float(s)==16 or s==98.14:
            return -999
        else:
            return s

    def auto_cubicspline_interpolation(self,l, not_lower_than_min = True):
        data=dict(zip(list(range(1,len(l)+1)), l)) 
        clean_data=data.copy()
        for key in data:
            if (self.is_valid(clean_data[key])==False ):
                del clean_data[key]
        x=list(clean_data.keys())
        y=list(clean_data.values())
        cs = CubicSpline(x, y,bc_type="natural")
        xs = np.arange(1, len(l)+1, 1)
        if not_lower_than_min:
            min_y = min(y)
            o = []
            for  a in cs(xs):
                if a<min_y:
                    o.append(min_y)
                else:
                    o.append(round(a,2))
            return o
        else:
            return [round(a,2) for a in cs(xs)]

    def count_consecutive_false(self,l):
        count=0
        max=0
        for item in l:
            if item==False:
                count=count+1
                if count>max:
                    max=count
            else:
                count=0
        return max

    def GDD_calculator (self, start_date, end_date, day_df):
        start_date_dt = parse(start_date)
        end_date_dt = parse(end_date)
        GDD=0
        day_df['觀測時間(hour)'] = pd.to_datetime(day_df['觀測時間(hour)'])
        while start_date_dt<end_date_dt:
            day_slice = day_df[(day_df['觀測時間(hour)']>=start_date_dt) & (day_df['觀測時間(hour)']<start_date_dt+timedelta(days=1))]
            GDD += (day_slice['氣溫(℃)'].max() + day_slice['氣溫(℃)'].min())/2 -10
            start_date_dt += timedelta(days=1)
        return GDD
    #load weather station list
    def agr_get_sta_list(self, area_id=0, level_id=0):
        my_headers = {    
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "ja-JP,ja;q=0.9,zh-TW;q=0.8,zh;q=0.7,en-US;q=0.6,en;q=0.5",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "sec-ch-ua": "\"Chromium\";v=\"92\", \" Not A;Brand\";v=\"99\", \"Google Chrome\";v=\"92\"",
            "sec-ch-ua-mobile": "?0",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-requested-with": "XMLHttpRequest", #required
        }
        URI = 'https://agr.cwb.gov.tw/NAGR/history/station_day/get_station_name'
        area = ['', '北', '中', '南', '東'][area_id]
        level = ['', '新農業站'][level_id]
        r1 = requests.post(URI, data={'area':area, 'level':level}, headers = my_headers)
        sta_dict = json.loads(r1.text)
        df = pd.DataFrame(sta_dict)
        extract_df = df[['ID', 'Cname', 'Altitude', 'Latitude_WGS84', 'Longitude_WGS84', 'Address', 'StnBeginTime', 'stnendtime', 'stationlist_auto']]
        extract_df.columns=['站號', '站名', '海拔高度(m)', '緯度', '經度', '地址', '資料起始日期', '撤站日期', '備註']
        return extract_df

    def load_weather_station_list(self, include_agr_sta = True, include_suspended=False, gitubCached = True):
        if gitubCached:
            df = pd.read_csv('https://raw.githubusercontent.com/Raingel/weather_station_list/main/data/weather_sta_list.csv')
            df = df.drop('Unnamed: 0', axis = 1)
            if include_suspended == False:
                df = df[df['撤站日期'].isnull()]
            if include_agr_sta == False:
                df = df[df['備註'] != '新農業站']
            return df.reset_index(drop=True)
        else:
            #load from CWB
            raw = pd.read_html('https://e-service.cwb.gov.tw/wdps/obs/state.htm')
            weather_station_list = raw[0]
            if include_suspended:
                weather_station_list = weather_station_list.append(raw[1])
            #load from agri
            if include_agr_sta:
                weather_station_list = weather_station_list.append(self.agr_get_sta_list(level_id=1), ignore_index = True)
        return weather_station_list

    def find_cloest_sta (self, lat, lon, date, weather_station_list):
        for index, row in weather_station_list.iterrows():
            weather_station_list.loc[index, 'distance'] = distance((lat,lon), (row['緯度'],row['經度'])).km
            try:
                if parse(row['資料起始日期'])> parse(date):
                    weather_station_list.loc[index, 'distance'] = 9999
            except:
                pass
        return weather_station_list.sort_values(by=['distance'])
    
    def min_max_normalization (self, x, min, max):
        return (x-min)/(max-min)

    def data_parse (self, x, parse_method=1):
    #for deep learning
        if parse_method==1:
            #feature selection and normalization
            #x_selected = x[['氣溫(℃)', '相對溼度(%)', '降水量(mm)', 'Wv', 'Wu', 'Month_normalized', 'Hour_normalized']].copy()
            x_selected = x[['氣溫(℃)', '相對溼度(%)', '降水量(mm)', 'Wv', 'Wu']].copy()
            x_selected['氣溫(℃)'] = (x_selected['氣溫(℃)']-22)/4.8
            x_selected['相對溼度(%)'] = (x_selected['相對溼度(%)']-80)/14
            x_selected['降水量(mm)'] = (x_selected['降水量(mm)']-0.11)/1.08
            x_selected['Wv'] = (x_selected['Wv']+0.6)/1.5
            x_selected['Wu'] = (x_selected['Wu']-0.17)/1.2

            #convert hour to day daya
            count=0
            new_output = []
            while count*24 < len(x_selected):
                x_day = x_selected[count*24:(count+1)*24]
                new_output.append([
                            x_day['氣溫(℃)'].max(),
                            x_day['氣溫(℃)'].mean(),
                            x_day['氣溫(℃)'].min(),
                            x_day['相對溼度(%)'].mean(),
                            x_day['降水量(mm)'].sum(),
                            x_day['Wv'].mean(),
                            x_day['Wu'].mean(),
                                ])
                count+=1
            return np.array(new_output)
            #return np.array(new_output)
        if parse_method == 2:
            #feature selection and normalization
            #x_selected = x[['氣溫(℃)', '相對溼度(%)', '降水量(mm)', 'Wv', 'Wu', 'Month_normalized', 'Hour_normalized']].copy()
            x_selected = x[['觀測時間(hour)','氣溫(℃)', '相對溼度(%)', '降水量(mm)', 'Wv', 'Wu']].copy()
            x_selected['氣溫(℃)'] = (x_selected['氣溫(℃)']-22)/4.8
            x_selected['相對溼度(%)'] = (x_selected['相對溼度(%)']-80)/14
            x_selected['降水量(mm)'] = (x_selected['降水量(mm)']-0.11)/1.08
            #x_selected['降水量(mm)'] = x_selected['降水量(mm)']/30
            x_selected['Wv'] = (x_selected['Wv']+0.6)/1.5
            x_selected['Wu'] = (x_selected['Wu']-0.17)/1.2

            x_day = ps.sqldf ("""SELECT MAX(`氣溫(℃)`), AVG(`氣溫(℃)`), MIN(`氣溫(℃)`), (MAX(`氣溫(℃)`) - MIN(`氣溫(℃)`)), AVG(`相對溼度(%)`),
                              MIN(`相對溼度(%)`), SUM(IIF(`相對溼度(%)`>0.714,1,0)), MAX(`降水量(mm)`), SUM(IIF(`降水量(mm)`>-0.01,1,0))
                              FROM x_selected GROUP BY DATE(`觀測時間(hour)`) 
                              ORDER BY DATE(`觀測時間(hour)`) ASC                              
                              """)
            
            return np.array(x_day.to_numpy())
            #return np.array(new_output)            
    def WRF_data (self, lat=20.5, lon=121): 
        URI = "http://mycolab.pp.nchu.edu.tw/WRF_APCP/WRF_API.php?lat={}&lon={}".format(lat, lon)
        #print (URI)
        df = pd.read_csv(URI)
        outputDf = pd.DataFrame()
        if df.empty != True:
            df['Valid'] = pd.to_datetime(df['Valid']) 
            outputDf['觀測時間(hour)'] = pd.date_range(df['Valid'].min(),df['Valid'].max(),freq='1H')
            outputDf = outputDf.merge(df,left_on='觀測時間(hour)',right_on='Valid', how='left')
            outputDf = outputDf.drop(['Initial','Valid'], axis=1)
            outputDf['Lon']= outputDf['Lon'].interpolate()
            outputDf['Lat']= outputDf['Lat'].interpolate()
            outputDf['APCP']= outputDf['APCP'].interpolate()
            outputDf['TMP']= outputDf['TMP'].interpolate(method = 'cubicspline') - 273.15
            outputDf['RH']= outputDf['RH'].interpolate(method = 'cubicspline')
            outputDf['NSWRS']= outputDf['NSWRS'].interpolate()
            outputDf['VGRD']= outputDf['VGRD'].interpolate(method = 'cubicspline')
            outputDf['UGRD']= outputDf['UGRD'].interpolate(method = 'cubicspline')
            outputDf['觀測時間(hour)'] = outputDf['觀測時間(hour)']+pd.Timedelta("8H") #from UTC to UTC+8
        return outputDf

    def WRFtoCODISformat (self,WRF):
        WRF['氣溫(℃)'] = WRF['TMP']
        WRF['相對溼度(%)'] = WRF['RH']
        WRF['降水量(mm)'] = WRF['APCP'] - ([0]+WRF['APCP'].tolist()[:-1])
        WRF['Wv'] = WRF['VGRD']
        WRF['Wu'] = WRF['UGRD']
        WRF['風速(m/s)'] = np.sqrt(WRF['UGRD']**2+WRF['VGRD']**2)
        wind_dir_trig_to = np.arctan2(WRF['UGRD']/WRF['風速(m/s)'], WRF['VGRD']/WRF['風速(m/s)']) 
        wind_dir_trig_to_degrees = wind_dir_trig_to * 180/3.1415926 ## -111.6 degrees
        wind_dir_trig_from_degrees = wind_dir_trig_to_degrees + 180 ## 68.38 degrees
        wind_dir_cardinal = 90 - wind_dir_trig_from_degrees
        WRF['風向(360degree)'] = wind_dir_cardinal
        WRF['全天空日射量(MJ/㎡)'] = WRF['NSWRS']*0.0036
        WRF['日照時數(hr)'] = WRF['全天空日射量(MJ/㎡)'].apply(lambda x: 1.0 if x/0.432 > 1 else x/0.432)
        return WRF[['觀測時間(hour)', '氣溫(℃)', '相對溼度(%)', '降水量(mm)', 'Wv', 'Wu', '風速(m/s)',
       '風向(360degree)', '全天空日射量(MJ/㎡)','日照時數(hr)']]

    def history_and_forecast(self, start_date = '20220302', sta_no = '466900'):
        end_date = datetime.today().strftime('%Y%m%d')
        past = self.get_weather_data(sta_no, start_date, end_date, interpolation = True)
        if past.empty:
            return pd.DataFrame()
        past.index = past['觀測時間(hour)']
        staList = self.load_weather_station_list()
        #Find the longitude and latitude of the station
        staLat = staList[staList['站號']==sta_no]['緯度'].values[0]
        staLon = staList[staList['站號']==sta_no]['經度'].values[0]
        WRF = self.WRF_data(staLat,staLon)
        if WRF.empty or past.empty:
            print('No data')
            return pd.DataFrame
        WRFCodis = self.WRFtoCODISformat(WRF)
        WRFCodis.index = WRFCodis['觀測時間(hour)']
        newdf = pd.DataFrame(columns=['觀測時間(hour)', '測站氣壓(hPa)', '海平面氣壓(hPa)', '氣溫(℃)', '露點溫度(℃)', '相對溼度(%)',
            '風速(m/s)', '風向(360degree)', '最大陣風(m/s)', '最大陣風風向(360degree)', '降水量(mm)',
            '降水時數(hr)', '日照時數(hr)', '全天空日射量(MJ/㎡)', '能見度(km)', '紫外線指數', '總雲量(0~10)',
            'Wv', 'Wu', 'Month_normalized', 'Hour_normalized'])
        newdf['觀測時間(hour)'] = pd.date_range(past['觀測時間(hour)'].min(),WRFCodis['觀測時間(hour)'].max(),freq='1H')
        newdf.index = pd.date_range(past['觀測時間(hour)'].min(),WRFCodis['觀測時間(hour)'].max(),freq='1H')
        newdf.update(past)
        newdf.update(WRFCodis)
        newdf['氣溫(℃)'] = pd.to_numeric(newdf['氣溫(℃)']).interpolate(method='linear')
        newdf['相對溼度(%)'] = pd.to_numeric(newdf['相對溼度(%)']).interpolate(method='linear')
        newdf['Wv'] = pd.to_numeric(newdf['Wv']).interpolate(method='linear')
        newdf['Wu'] = pd.to_numeric(newdf['Wu']).interpolate(method='linear')
        newdf['降水量(mm)'] = pd.to_numeric(newdf['降水量(mm)']).interpolate(method='linear')
        newdf['全天空日射量(MJ/㎡)'] = pd.to_numeric(newdf['全天空日射量(MJ/㎡)']).interpolate(method='linear')
        newdf['日照時數(hr)'] = pd.to_numeric(newdf['日照時數(hr)']).interpolate(method='linear')
        newdf['風速(m/s)'] = np.sqrt(newdf['Wv']**2+newdf['Wu']**2)
        wind_dir_trig_to = np.arctan2(newdf['Wu']/newdf['風速(m/s)'], newdf['Wv']/newdf['風速(m/s)']) 
        wind_dir_trig_to_degrees = wind_dir_trig_to * 180/3.1415926 ## -111.6 degrees
        wind_dir_trig_from_degrees = wind_dir_trig_to_degrees + 180 ## 68.38 degrees
        wind_dir_cardinal = 90 - wind_dir_trig_from_degrees
        newdf['風向(360degree)'] = wind_dir_cardinal
        return newdf[['觀測時間(hour)', '氣溫(℃)', '相對溼度(%)','降水量(mm)', 'Wv', 'Wu','風速(m/s)', '全天空日射量(MJ/㎡)','日照時數(hr)']]

# %%
if __name__ == '__main__':
    from matplotlib import rcParams
    tW = taiwanWeather()
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']
    df = tW.get_weather_data('466900', '20220302', '20220802', interpolation = True, type='daily')

# %%



