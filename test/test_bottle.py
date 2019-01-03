import bottle
from bottle import route,run

@route('/')
def index():
    return 'hello, bottle'
run(host='127.0.0.1', port=22333)
app = bottle.Bottle()
