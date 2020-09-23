# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:09:53 2019

@author: ashish soni
"""
from __future__ import print_function
import time
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from random import random
import os
import requests
import json
    
    


    
def TranscribeRecordedSpeech(FILE_DIR,FILE_NAME,option):
    if(option=='AWS'): 
        
        ACCESS_KEY_ID = r.smt_AWS_ACCESS_KEY_ID
        ACCESS_SECRET_KEY = r.smt_AWS_ACCESS_SECRET_KEY 
        AWS_REGION = r.smt_AWS_REGION
        BUCKET_NAME = r.smt_AWS_BUCKET_NAME  
        upload_status = VoiceFile_AWSS3Upload(FILE_DIR,
                                              FILE_NAME,
                                              ACCESS_KEY_ID,
                                              ACCESS_SECRET_KEY,
                                              AWS_REGION,
                                              BUCKET_NAME)
        print('upload_status: ',upload_status)
        if(upload_status==True): 
            speech2text,transcribe_status = VoiceFile_AWSTranscribe(FILE_NAME,
                                                                    ACCESS_KEY_ID,
                                                                    ACCESS_SECRET_KEY,
                                                                    AWS_REGION,
                                                                    BUCKET_NAME)
        else:
            transcribe_status='TRANSCRIPTION FAILED'
    
    if(option=='NTE'):
        speech2text,transcribe_status = VoiceFile_NuanceTranscribe(FILE_DIR,FILE_NAME) 
               
    FILE_txt = FILE_DIR+'/'+ FILE_NAME[0:-4]+'.txt'
    file = open(FILE_txt, 'w')
    # write text
    file.write(speech2text)
    # close file
    file.close()
    return(speech2text,transcribe_status)
    
    
    
    
### 2 nuance
def VoiceFile_NuanceTranscribe(FILE_DIR,FILE_NAME):

    #speech2text,transcribe_status = VoiceFile_NuanceTranscribe(FILE_DIR,FILE_NAME)  
    # POST with JSON 

    job_url = 'http://localhost:8080/v4/jobs'
    headers = {'Content-type': 'application/json'}
    
    FILE = 'file:///'+FILE_DIR+'/'+FILE_NAME
    
    payload = {"job_type":"batch_transcription", 
                "operating_mode":"accurate", 
                 "model":{ 
                     "name":"eng-GBR"
                  },
                  
                "channels":{ 
                    "channel1": { 
                        "url":FILE, 
                        "format":"audio/wave", 
                        "result_format":"transcript",
                        "diarize": True
                        # "num_speakers": "2"
                     } 
                             
                } 
    }

    r = requests.post(job_url, json=payload,headers=headers)
    #print(r.json())
    
    job_results_url = job_url+'/'+r.json()['reference']+'/results'
    #result = requests.get(job_results_url)
    while True: 
        result = requests.get(job_results_url) 
        time.sleep(5)
        if result.json()['errors'] == [] :
            #print('exiting while loop')
            break
            
    transcribe_status = result.json()['status']
    
    lst = result.json()['channels']['channel1']['transcript']
    
    tmp = ''
    transcript=[]
    curr_spkr=lst[0]['speaker']
    prev_spkr =lst[0]['speaker']
    
    for n in range(len(lst)):
        curr_spkr = lst[n]['speaker']
        if curr_spkr == prev_spkr :
            tmp = tmp + ' '+ lst[n]['text'] 
            
        else:
            transcript.append('spk_'+ prev_spkr + ': ' + tmp) 
            tmp = lst[n]['text']
            
        prev_spkr = curr_spkr
            
    transcript.append('spk_'+ prev_spkr + ': ' + tmp)   
    transcribed_text = '\n'.join(transcript)    
    
    return(transcribed_text,transcribe_status)
        
def Increment_SpeakerIndex_InSpeakerLabel(label):
    return 'spk_'+str(int(label.replace("spk_", ""))+1)
       
def VoiceFile_AWSS3Upload(FILE_DIR,
                          FILE_NAME,
                          ACCESS_KEY_ID,
                          ACCESS_SECRET_KEY,
                          AWS_REGION,
                          BUCKET_NAME):
 
    FILE = FILE_DIR+'/'+FILE_NAME
    data = open(FILE, 'rb')
    
    print("Voice File Opened...")
    
    # S3 Connect
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        region_name = AWS_REGION,
        config=Config(signature_version='s3v4')
    )
    
    print("Voice File upload started...")
    # file Uploaded
    s3.Bucket(BUCKET_NAME).put_object(Key=FILE_NAME, Body=data)

    def check(s3, bucket, key):
        try:
            s3.Object(bucket, key).load()
        except ClientError as e:
            return int(e.response['Error']['Code']) != 404
        return True
       
    upload_status = check(s3, BUCKET_NAME, FILE_NAME)
    
    print("Voice File upload status : ",upload_status)
    
    return(upload_status)
    

def VoiceFile_AWSTranscribe(FILE_NAME,
                            ACCESS_KEY_ID,
                            ACCESS_SECRET_KEY,
                            AWS_REGION,
                            BUCKET_NAME):
    
    # S3 Connect
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        region_name = AWS_REGION,
        config=Config(signature_version='s3v4')
    )
    
    
    sess = boto3.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        region_name = AWS_REGION)
    
    transcribe = sess.client('transcribe')
    
    #job_uri = "https://smt0001.s3.eu-west-2.amazonaws.com/"+ FILE_NAME
    
    job_uri = "https://"+BUCKET_NAME+".s3."+AWS_REGION+".amazonaws.com/"+ FILE_NAME
    
    #job_uri = "https://s3.amazonaws.com/smt002/"+ FILE_NAME
    
    print("Started Voice Transcription ...")
    
    job_name = 'job-'+time.strftime("%m%d%Y-%H%M%S", time.localtime())
    #job_name = 'a001'
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-GB',
        Settings= { 
             #"ChannelIdentification": True,
             "MaxSpeakerLabels": 2,
             "ShowSpeakerLabels": True
          }
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            print("Result : TRANSCRIPTION ",status['TranscriptionJob']['TranscriptionJobStatus'])
            break
        print("Not ready yet..")
        time.sleep(5)
    #print(status)
    import urllib.request
    
    url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    webURL = urllib.request.urlopen(url)
    data = webURL.read()
    JSON_object = json.loads(data.decode('utf-8'))
    
    transcribe_status = status['TranscriptionJob']['TranscriptionJobStatus']
    
    #text = JSON_object['results']['transcripts'][0]['transcript']
    
    text = JSON_object['results']
    
    listofsegments = []
    for seg in range(len(text['speaker_labels']['segments'])):
        if(text['speaker_labels']['segments'][seg]['speaker_label']!=''):
            listofsegments.append(seg)
            
    transcript = []
    for segment_no in listofsegments:
        for l in range(len(text['items'])):
            if 'end_time' in text['items'][l].keys():
                if text['items'][l]['end_time'] == text['speaker_labels']['segments'][segment_no]['end_time']:
                    end = l
                    break;
    
        for l in range(len(text['items'])):
            if 'start_time' in text['items'][l].keys():
                if text['items'][l]['start_time'] == text['speaker_labels']['segments'][segment_no]['start_time']:
                    start = l
                    break;
        txt = Increment_SpeakerIndex_InSpeakerLabel(text['speaker_labels']['segments'][segment_no]['speaker_label']) + ': '
        for k in range(start,end+2):
            txt = txt + text['items'][k]['alternatives'][0]['content']+ ' '
        transcript.append(txt)
    
    transcribed_text = '\n'.join(transcript)

    return(transcribed_text, transcribe_status)
