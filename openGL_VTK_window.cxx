/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#include "cpp_headers.h"
#include "vtk_headers.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "openGL_VTK_window.h"

using namespace std;

vtkNew<vtkConeSource> cone;

openGL_VTK_window :: openGL_VTK_window(QWidget *parent)
 : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{

    vtkNew<vtkPolyDataMapper> coneMapper;
    vtkNew<vtkActor> coneActor;
    coneMapper->SetInputConnection( cone->GetOutputPort() );
    coneActor->SetMapper( coneMapper.GetPointer() );

    ren->AddActor( coneActor.GetPointer() );
    renWin->AddRenderer( ren.GetPointer() );

    this->resize(800, 600);
}

void openGL_VTK_window :: initializeGL()
{
    // Here is the trick: we ask the RenderWindow to join the current OpenGL context created by GLUT
    renWin->InitializeFromCurrentContext();
    ren->EraseOff(); // important!
    ren->LightFollowCameraOn();
    ren->TwoSidedLightingOn();

    cout << "init" << endl;
}
void openGL_VTK_window :: resizeGL(int w, int h)
{
    renWin->SetSize( w, h );
    cout << "resize" << endl;
}

void openGL_VTK_window :: keyPressEvent(QKeyEvent* QE)
{
}


void openGL_VTK_window :: mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
    if (event->button() == Qt::LeftButton)
        xform_mode = XFORM_ROTATE;
    else if (event->button()==Qt::RightButton)
        xform_mode = XFORM_SCALE;
    else
        xform_mode = XFORM_NONE;

    cout<<"clicked "<<event->x()<<"  "<<event->y()<<endl;
}

void openGL_VTK_window :: mouseMoveEvent(QMouseEvent *event)
{
    cout<<"dragging "<<event->x()<<"  "<<event->y()<<endl;

    if (xform_mode==XFORM_ROTATE) {
      float x_angle = (event->x() - lastPos.x())/2;
      //if (x_angle > 180) x_angle -= 360;
      //else if (x_angle <-180) x_angle += 360;

      float y_angle = (event->y() - lastPos.y())/2;

      double axis[3];
      axis[0] = -y_angle;
      axis[1] = -x_angle;
      axis[2] = 0;
      double mag = (y_angle*y_angle+x_angle*x_angle);
      transform->RotateWXYZ(mag, axis);
    }
    else if (xform_mode == XFORM_SCALE){
      float scale_size = (1 - (event->y() - lastPos.y())/120.0);
      if (scale_size <1e-5) scale_size = 1e-5;
      transform->Scale(scale_size, scale_size, scale_size);
    }
    lastPos = event->pos();
    this->update();
}

void openGL_VTK_window :: paintGL()
{
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_NICEST);
    glEnable(GL_LINE_SMOOTH);

    //glutSolidTeapot(1);
    glBegin(GL_POINTS);
    glColor3f(1,1,1);
    glVertex2f(0,0);
    glVertex2f(1,0);
    glVertex2f(0,1);
    glEnd();
    glPointSize(10);

    cout << "paint" << endl;

    // vtk camera -> set opengl modelview matrix
    vtkCamera *camera = ren->GetActiveCamera();
    double eyein[3] = {0,0,5};
    double upin[3] = {0, 1, 0};
    camera->SetPosition(transform->TransformVector(eyein));
    camera->SetViewUp(transform->TransformVector(upin));
    camera->SetFocalPoint(0,0,0);

    // get modelview matrix
    vtkMatrix4x4 *m = camera->GetModelViewTransformMatrix();
    double *f = &m->Element[0][0];

    // opengl
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultTransposeMatrixd(f);

    //////////// draw /////////
    opengl_draw();
    ///////////////////////////


    glLoadIdentity();
    renWin->Render();


    //glutSwapBuffers();
}

void openGL_VTK_window::opengl_draw()
{

}
