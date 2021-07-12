from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page

def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            cement=float(request.form['cement'])
            blast_furnance_slag = float(request.form['blast_furnance_slag'])
            fly_ash = float(request.form['fly_ash'])
            water_component = float(request.form['water_component'])
            superplasticizer = float(request.form['superplasticizer'])
            coarse_aggregate = float(request.form['coarse_aggregate'])
            fine_aggregate = float(request.form['fine_aggregate'])
            age_day = int(request.form['age_day'])

            value=[cement, blast_furnance_slag, fly_ash, water_component, superplasticizer, coarse_aggregate, fine_aggregate, age_day]

            l2 = []
            for x in value:
                x = x + 1
                x = np.log(x)
                l2.append(x)


            cluster_filename = 'kmeans_clustering.pickle'
            kmeans_model = pickle.load(open(cluster_filename, 'rb'))  # loading the model file from the storage
            # predictions using the loaded model file
            cluster_prediction = kmeans_model.predict([l2])

            model_filename = 'Model_' + str(cluster_prediction[0]) + '.pickle'
            loaded_model = pickle.load(open(model_filename, 'rb'))
            prediction = loaded_model.predict([l2])

            del(l2)
            return render_template('results.html', prediction=prediction[0])

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":

	app.run(debug=True)
