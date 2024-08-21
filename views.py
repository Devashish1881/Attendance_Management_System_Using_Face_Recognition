from django.shortcuts import render
from .models import Student, Attendance
import cv2
import numpy as np
import face_recognition

def register_student(request):
    if request.method == 'POST':
        name = request.POST['name']
        roll_number = request.POST['roll_number']
        image = request.FILES['image']

        # Load the image using OpenCV
        img = face_recognition.load_image_file(image)
        face_encoding = face_recognition.face_encodings(img)[0]

        student = Student(name=name, roll_number=roll_number, face_encoding=face_encoding)
        student.save()

        return render(request, 'attendance/register_success.html', {'student': student})

    return render(request, 'attendance/register_student.html')

def take_attendance(request):
    students = Student.objects.all()
    attendance_records = []

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([student.face_encoding for student in students], face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = students[match_index].name

                attendance, created = Attendance.objects.get_or_create(student=students[match_index])
                attendance.status = 'Present'
                attendance.save()
                attendance_records.append(attendance)

        if len(attendance_records) >= len(students):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return render(request, 'attendance/attendance_results.html', {'attendance_records': attendance_records})
