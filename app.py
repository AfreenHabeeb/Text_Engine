from flask import render_template,Flask,request,redirect,jsonify
from text_reader_writer import text_read_write
from werkzeug import secure_filename
import pandas as pd
import os
from sqlalchemy import create_engine

# from config import fnInitConfig

# spark = fnInitConfig("", "")
# print('---> Spark Session Created. Good to go...!!!')

engine = create_engine('mysql+pymysql://root:root@localhost:3306/vz_analytics', echo=True)

#Initializing Flask
app = Flask(__name__,template_folder='template')

@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html', title='home')

@app.route('/configparams')
def configparams():
    return render_template('configparams.html')

@app.route('/getparams',methods=['GET', 'POST'])
def getparams():
    inputDataFormat=request.form.get('inputDataFormat')
    inputDBname = request.form.get('inputDBname')
    inputTablename = request.form.get('inputTablename')
    inputFileNameWithPath = request.form.get('inputFileNameWithPath')
    inputDataSeparator = request.form.get('inputDataSeparator')
    inputHeader = request.form.get('inputHeader')
    generateUniqueID = request.form.get('generateUniqueID')
    groupByID = request.form.get('groupByID')
    outputDataFormat = request.form.get('outputDataFormat')
    outputFilesLocation = request.form.get('outputFilesLocation')
    repoDirectoryPath = request.form.get('repoDirectoryPath')
    runSentiment = request.form.get('runSentiment')
    runNgram = request.form.get('runNgram')
    requireTopics = request.form.get('requireTopics')
    optimizeTopics = request.form.get('optimizeTopics')
    topicOnGrams = request.form.get('topicOnGrams')
    u_numOfTopics = request.form.get('u_numOfTopics')
    outputDBname = request.form.get('outputDBname')
    outputSentimentScoreTable = request.form.get('outputSentimentScoreTable')
    outputNgramsTable = request.form.get('outputNgramsTable')
    prodRun = request.form.get('prodRun')

    print([inputDataFormat, inputDBname, inputTablename, inputFileNameWithPath, inputDataSeparator, inputHeader, generateUniqueID,
           groupByID,outputDataFormat, outputFilesLocation, repoDirectoryPath, runSentiment, runNgram, requireTopics,
           optimizeTopics, topicOnGrams, u_numOfTopics, outputDBname, outputSentimentScoreTable, outputNgramsTable, prodRun])

    path="C:/Users/HABEEAF/Desktop/Afreen/TextEngine_python_v2/TextEngine/conf/TE_configuration_params_v1.txt"

    result=text_read_write(path=path,
                           inputDataFormat=inputDataFormat, inputDBname=inputDBname, inputTablename=inputTablename,
                           inputFileNameWithPath=inputFileNameWithPath, inputDataSeparator=inputDataSeparator,
                           inputHeader=inputHeader, generateUniqueID=generateUniqueID,groupByID=groupByID,
                           outputDataFormat=outputDataFormat, outputFilesLocation=outputFilesLocation,
                           repoDirectoryPath=repoDirectoryPath, runSentiment=runSentiment, runNgram=runNgram,
                           requireTopics=requireTopics, optimizeTopics=optimizeTopics, topicOnGrams=topicOnGrams,
                           u_numOfTopics=u_numOfTopics, outputDBname=outputDBname,
                           outputSentimentScoreTable=outputSentimentScoreTable, outputNgramsTable=outputNgramsTable,
                           prodRun=prodRun)

    from main import function_call
    exit_code = function_call(path)

    if exit_code == '000':
        return jsonify({'message': "HR-Exit Text engine Ran Successfully"})
    else:
        return jsonify({'message': "HR-Exit Text engine Ran Failed"})

    return jsonify({'message': result})

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        file_val = request.files['file']
        return jsonify({'message': 'success!'})

@app.route('/uploader', methods=['POST'])
def uploader():
    df=pd.DataFrame()
    if request.method == 'POST':
      try:
        f = request.files['file']
        # print("req",request)
        # print("---",request.files['file'])
        f.save('uploaded_files/'+secure_filename(f.filename))
        name=secure_filename(f.filename)
        # print(name)
      #   stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv('uploaded_files/'+f.filename, encoding='cp1252')
        fname=os.path.splitext(name)
        df.to_sql(fname[0], con=engine)
        return jsonify({'message': "Table created successfully"})
      except Exception as e:
        print(e)
        return jsonify({'message': "Table already exist or some other error occur in uploader()"})

if __name__ == '__main__':
   app.run(debug=True)
