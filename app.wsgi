#!/var/www/FLASKAPPS/airseadb/env-airseadb/bin/python3    #/usr/bin/python
import sys
sys.path.insert(0,"/var/www/FLASKAPPS")
#print(sys.prefix)

#activate_this = '/var/www/FLASKAPPS/airseadb/venv/bin/activate_this.py'
#with open(activate_this) as file_:
#    exec(file_.read(), dict(__file__=activate_this))

from airseadb.app import server as application

