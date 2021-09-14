import numpy as np
import requests
from bs4 import BeautifulSoup
from xml.etree.ElementTree import XML, fromstring, tostring
from datetime import datetime
import pandas as pd 
import sys
import matplotlib.pyplot as plt


class HERE_Traffic():
    def __init__(self):
        # collect current datetime
        now = datetime.now()
        self.timestamp = now.strftime("%Y%m%d%H%M")
    
    def credentials(self, APP_ID, APP_CODE):
        '''Please enter HERE API credentials:

        API_ID : Enter HERE account application identification.
        APP_CODE : Enter HERE account application code.
        API_KEY: Enter ONLY for traffic incident data. Default value is null.
        '''
        self.APP_ID = APP_ID
        self.APP_CODE = APP_CODE

    def bbox(self, lat0, lon0, lat1, lon1):
        '''Bounding-box information for traffic flow.
        '''
        self.lat0 = lat0
        self.lon0 = lon0
        self.lat1 = lat1
        self.lon1 = lon1

    def _connect_traffic_flow(self):
        '''Connects to HERE Traffic API using bbox and API credentials. Returns parsed XML response.
        If invalid credentials are used returns error message.
        '''
    
        page = requests.get(f'https://traffic.api.here.com/traffic/6.3/flow.xml?app_id={self.APP_ID}&app_code={self.APP_CODE}&bbox={self.lat0},{self.lon0};{self.lat1},{self.lon1}&responseattributes=sh,fc')
        
        # if credentials are incorrect, prompt error message and kill program.
        if str(page) == '<Response [401]>':
            sys.exit("Invalid credentials. Please re-enter correct API identification number or code.")
        else:
            soup = BeautifulSoup(page.text, "lxml")
            response = soup.find_all('fi')

        return response
    
    def _connect_incident_reports(self):
        '''Connects to HERE Incident API using bbox and API credentials. Returns parsed html response.
        '''
        page = requests.get(f'https://traffic.api.here.com/traffic/6.3/incidents.xml?app_id={self.APP_ID}&app_code={self.APP_CODE}&bbox={self.lat0},{self.lon0};{self.lat1},{self.lon1}&responseattributes=sh,fc')
        soup = BeautifulSoup(page.text, "html.parser")
        parsed = soup.find_all("trafficml_incidents")[0]

        return parsed


    def traffic_flow(self):
        '''
        Returns traffic flow information, latitudes, longitudes, road names. 
        '''
        # call traffic response 
        response = self._connect_traffic_flow()
        # loop through each road and collect road type, road speed limit, actual real-time 
        # road speed, road name, and traffic direction. 
        a1=[]
        loc_list_hv=[]
        lats=[]
        lons=[]
        speed_uncapped=[]
        speed_capped=[]
        jam_factor=[]
        free_flow_spd=[]
        names = []
        direction=[]
        c=0
        for html_response in response:
            #for j in range(0,len(shps)):
            xml_response = fromstring(str(html_response))
            fc=5
            for road in xml_response:
                if('fc' in road.attrib):
                    fc=int(road.attrib['fc'])
                if('cn' in road.attrib):
                    cn=float(road.attrib['cn'])
                if('su' in road.attrib):
                    su=float(road.attrib['su'])
                if('sp' in road.attrib):
                    sp=float(road.attrib['sp'])
                if('jf' in road.attrib):
                    jf=float(road.attrib['jf'])
                if('ff' in road.attrib):
                    ff=float(road.attrib['ff'])
                if('de' in road.attrib):
                    de=(road.attrib['de'])
                if('qd' in road.attrib):
                    qd=(road.attrib['qd'])
                
            # split road information into individual latitude/longitude arrays based on road shape
            # fc is highways and major roadways. CN is confidence in real-time traffic flow information (max 1). At least 70%.
            if((fc<=5) and (cn>=0.7)):

                # road shapes by lat/lon coordinates
                shps=html_response.find_all("shp")

                for j in range(0,len(shps)):
                    latlong=shps[j].text.replace(',',' ').split()
                    #loc_list=[]
                    la=[]
                    lo=[]
                    su1=[]
                    ff1=[]
                    sp1=[]
                    jf1=[]
                    qd1 = []
                    name=[]

                    for i in range(0,int(len(latlong)/2)):
                        # organized as pairs (lat , lon) therefore split by lat/lon values. 
                        loc_list_hv.append([float(latlong[2*i]),float(latlong[2*i+1]),float(su),float(ff)])
                        la.append(float(latlong[2*i]))
                        lo.append(float(latlong[2*i+1]))
                        su1.append(float(su))
                        ff1.append(float(ff))
                        sp1.append(float(sp))
                        jf1.append(float(jf))
                        name.append(de)
                        qd1.append(qd)
                    lats.append(la)
                    lons.append(lo)
                    speed_uncapped.append(np.mean(su1))
                    speed_capped.append(np.mean(sp1))
                    jam_factor.append(np.mean(jf1))
                    free_flow_spd.append(np.mean(ff1))
                    names.append(str(de))
                    direction.append(qd1[0])
                    
        return names, direction, lats, lons, speed_capped ,speed_uncapped, free_flow_spd, jam_factor

    def incident_report(self):
        ''' Returns incident type (construction, roadwork, accident, etc) along with coordinates of the incident location and the roads affected.
        '''
        # call incident response 
        response = self._connect_incident_reports()

        # collect the incident description, status, coordinates, and street path coordinates
        descs = response.find_all("traffic_item_type_desc")
        stat = response.find_all("traffic_item_status_short_desc")
        point_lat = response.find_all("latitude")
        point_lon = response.find_all("longitude")
        shps = response.find_all("shapes")

        # predefine variables
        lats=[]
        lons=[]
        latlong = []
        descriptions = []
        status = []
        latitude = []
        longitude = []

        # loop and collect the incident information for each case
        for j in range(0,len(shps)):
            latlong=shps[j].text.replace(',',' ').split()
            desc = descs[j].text
            stats = stat[j].text
            point_lats = point_lat[j].text
            point_lons = point_lon[j].text
            la=[]
            lo=[]
            # remove combined latlon values that are duplicates and incorrectly listed
            [latlong.remove(x) for x in latlong if len(x)>10]

            # assign each lat and lon point
            for i in range(0,int(len(latlong)/2)):
                la.append(float(latlong[2*i]))
                lo.append(float(latlong[2*i+1]))  
            
            # append the information into lists 
            lats.append(la)
            lons.append(lo)
            descriptions.append(desc) 
            status.append(stats) 
            latitude.append(point_lats) 
            longitude.append(point_lons)  

        return descriptions, status, latitude, longitude, lats, lons

    def generate_traffic_csv(self, outdir):
        '''This function generates a CSV containing traffic variables for each road segment in the specified bbox.
        Please specify an out directory.
        '''
        # call variables from traffic flow function
        names, qd, lats, lons, speed_capped ,speed_uncapped, free_flow_spd, jam_factor = self.traffic_flow()

        # store variables in dataframe
        traffic_df = pd.DataFrame({"road_name": names, "direction":qd,"lats":lats, "lons":lons, "sp":speed_capped, "su":speed_uncapped,"ffs":free_flow_spd, "jf":jam_factor})
        traffic_df.sp = traffic_df.sp.round(0)
        traffic_df.su = traffic_df.su.round(0)
        traffic_df.ffs = traffic_df.ffs.round(0)

        # create CSV file containing the variables
        traffic_df.to_csv(f"{outdir}/oxford_traffic_{self.timestamp}.csv")

    def generate_incident_csv(self, outdir):
        '''This function generates a CSV containing incident variables for each road segment in the specified bbox.
        Please specify an out directory.
        '''
        # call variables from incident report function
        descriptions, status, latitude, longitude, lats0, lons0 = self.incident_report()

        # store variables in dataframe
        incident_df = pd.DataFrame({"status":status, "desc":descriptions, "point_lat":latitude, "point_lon":longitude,
        "lats":lats0, "lons":lons0})

        # create CSV file containing the variables
        incident_df.to_csv(f"{outdir}/oxford_incident_{self.timestamp}.csv")




