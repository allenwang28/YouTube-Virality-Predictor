import csv
import requests
import urllib2
from lxml import html, etree
import json

# Looking at top YouTube Influencers
# https://www.youtube.com/channel/UC1nC1_rOpVnQCc2F2lDHx0Q
# Fitness "UC1nC1_rOpVnQCc2F2lDHx0Q"
# CGuzman "UCU1iJ2ChGwaNLvBjip0p2Ag"

api_key = 'AIzaSyCrFWiPfGcb5IsyS-wpAMk6eaNdMaC8pXs'
channel = 'UCF0pVplsI8R5kcAqgtoRqoA'
vidStats = 'https://www.googleapis.com/youtube/v3/videos?part=id,statistics&id='
vidSnips = 'https://www.googleapis.com/youtube/v3/videos?part=id,snippet&id='

channelID = str(input("Enter Channel ID: "))

channelSearch = 'https://www.googleapis.com/youtube/v3/search?key=' + api_key +  '&channelId=' + channelID + "&part=snippet,id&order=date&maxResults=50"

print(channelSearch)

stats = json.load(urllib2.urlopen(channelSearch))

print(stats)