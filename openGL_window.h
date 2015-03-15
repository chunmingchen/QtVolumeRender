/////////////////////////////////////////////////////
/// OpenGL example widget on Qt
/////////////////////////////////////////////////////


#ifndef OPENGL_HEADER_H
#define OPENGL_HEADER_H

#include <cpp_headers.h>
#include <QtOpenGL/QGLWidget>
#include <QMouseEvent>

class openGL_window : public QGLWidget
{
    Q_OBJECT

public:

    openGL_window(QWidget *parent);
    int window_id;  

    void paintGL();
    void initializeGL();
    void resizeGL(int w,int h);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *);
};

#endif
