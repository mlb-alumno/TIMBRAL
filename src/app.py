from flask import Flask, render_template, request, send_file, redirect, url_for
import interpolation
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audios')
def audios():
    return render_template('audios.html')

@app.route('/choose',methods=["GET","POST"])
def choose():
    if request.method == "POST":
         req = request.form
         path = "src/sound-selection/"
         sound1path =path + req["sound1path"].lower() +'.wav'
         sound2path =path + req["sound2path"].lower() +'.wav'
         
         interpolation.interpolate_2_audios_web_choose(sound1path,sound2path)

         return redirect(url_for('audios'))


@app.route('/generate')
def generate():    
    interpolation.interpolate_2_audios_web()
    return redirect(url_for('audios'))



@app.route('/download')
def download():
    return send_file("audio-tests/interpolation/generated-audios.zip",as_attachment=True)

@app.route('/audios/original1')
def original1():
        path = 'audio-tests/interpolation/generated/original/sound1.wav'
        return send_file(path,mimetype="audio/wav",as_attachment=True)
@app.route('/audios/original2')
def original2():
        path = 'audio-tests/interpolation/generated/original/sound2.wav'
        return send_file(path,mimetype="audio/wav",as_attachment=True)
@app.route('/audios/all')
def all():
        path = 'audio-tests/interpolation/generated/all.wav'
        return send_file(path,mimetype="audio/wav",as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)