from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello! this is main page <h1>Hello<h1>"


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    # TODO: set debug to False for production Heroku build
    app.run(debug=True, host="0.0.0.0", port=port)
