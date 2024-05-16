import os
from unicodedata import normalize


SUPPORTED_IMG_EXTS = ['jpg', 'jpeg', 'png']

def get_actor_name(file_path):
    dirname = os.path.basename(file_path)
    name = normalize_name(dirname)
    return name

def normalize_name(name):
    name = normalize('NFC', name)
    name = name.replace(' ', '_')
    name = name.lower()   
    return name

def extract_data(response, include_faces=False):
    data = {
        'face_embs' : [face.preds['verify'].logits.cpu() for face in response.faces],
        'bboxes': [face.loc for face in response.faces],
        'emotions': list(set([face.preds['fer'].label for face in response.faces]))
    }

    if include_faces:
        img = response.img
        face_imgs = []
        for face in response.faces:
            x1, y1, x2, y2 = face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2
            face_img = img[:, y1:y2, x1:x2]
            face_imgs.append(face_img)
        data['face_imgs'] = face_imgs

    return data

def to_serializable(response):
    data = {
        'face_embs' : [e.numpy().tolist() for e in response['face_embs']],
        'bboxes': [(loc.x1, loc.y1, loc.x2, loc.y2) for loc in response['bboxes']],
        'emotions': response['emotions']
    }
    if 'face_imgs' in response:
        data['face_imgs'] = [img.tolist() for img in response['face_imgs']]
        
    return data