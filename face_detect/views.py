from django.shortcuts import render


def index(request):
    return render(
        request,
        'face_detect/index.html'
    )

def home(request):
    return render(
        request,
        'face_detect/home.html'
    )
