# -*- coding: utf-8 -*-
from label_image import *

from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename

import tensorflow as tf

UPLOAD_FOLDER = 'path'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('home.html')
	if request.method == 'POST':
		file = request.files['file']
		# check if the post request has the file part
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_name = "path\\" + filename
		model_file = "retrained_graph.pb"
		label_file = "retrained_labels.txt"
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
		for i in top_k:
			if results[i] < 0.5:
				labels[i] = "No Result"
			return render_template('layout.html', answer=labels[i])

if __name__ == '__main__':
	app.run(debug = True)