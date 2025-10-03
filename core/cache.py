import hashlib


def key_for(defect, action):
text = f"{defect or ''}||{action or ''}".encode("utf-8","ignore")
return hashlib.sha1(text).hexdigest()
