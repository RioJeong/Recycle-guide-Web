
from label_image import *

from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename

import tensorflow as tf
from app.module import dbModule
UPLOAD_FOLDER = 'app\\static\\img\\path'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

from app.test.test import test as test
app.register_blueprint(test) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('home.html')
	if request.method == 'POST':
		file = request.files['file']
		# check if the post request has the file part
		filename = secure_filename(file.filename)

		static_img_path = "img/path/"+filename

		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_name = "app\\static\\img\\path\\" + filename
		model_file = "app\\retrained_graph.pb"
		label_file = "app\\retrained_labels.txt"
		input_height = 224
		input_width = 224
		input_mean = 128
		input_std = 128
		input_layer = "input"
		output_layer = "final_result"
		graph = load_graph(model_file)
		t = read_tensor_from_image_file(file_name,
			input_height=input_height,
			input_width=input_width,
			input_mean=input_mean,
			input_std=input_std)
		input_name = "import/" + input_layer
		output_name = "import/" + output_layer
		input_operation = graph.get_operation_by_name(input_name);
		output_operation = graph.get_operation_by_name(output_name);

		with tf.Session(graph=graph) as sess:
			start = time.time()
			results = sess.run(output_operation.outputs[0],
				{input_operation.outputs[0]: t})
			end=time.time()
		results = np.squeeze(results)

		answer = 0


		
		top_k = results.argsort()[-5:][::-1]
		labels = load_labels(label_file)
		cnt=0
		for i in top_k:
			if results[i] < 0.5:
				labels[i] = "No Result"	
			if cnt==0:
				answer=labels[i]
			cnt+=1

		db_class = dbModule.Database()

		sql      = "SELECT name, solution \
					FROM recycle.description \
					WHERE num=(SELECT num \
								FROM recycle.labeling \
								WHERE label=\"" + answer +"\")"
		row      = db_class.executeAll(sql)
		sql2 = "SELECT korean \
				FROM recycle.labeling \
				WHERE label = \"" + answer + "\""
		row2 = db_class.executeAll(sql2)
		return render_template('layout.html', answer=answer, resultData=row[0], image_file=static_img_path, korean = row2[0])



