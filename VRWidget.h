/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#ifndef VRWIDGE_HEADER_H
#define VRWIDGE_HEADER_H

#include <cpp_headers.h>
#include "CudaGLBase.h"
#include "openGL_VTK_window.h"
#include "vtk_headers.h"

#define TrFn_WIDTH 256

class VRWidget : public openGL_VTK_window
{
    Q_OBJECT

public:

    VRWidget(QWidget *parent);
    virtual ~VRWidget();

    //virtual void paintGL();
    virtual void initializeGL();
    virtual void resizeGL(int w,int h);
    //virtual void mousePressEvent(QMouseEvent *event);
    //virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void keyPressEvent(QKeyEvent *);


    void setData(vtkSmartPointer<vtkImageData> data);

    void setTrFn2D(std::vector<float4> &trfn)
    {
        g_uploadTrFn(&trfn[0], TrFn_WIDTH*TrFn_WIDTH);
    }

    void removeData() {//todo
        g_releaseTrFn();
    }

protected:
    CudaGLBase *gpuComm;

    bool initialized = false;

    virtual void opengl_draw();

    inline void setBgColor(unsigned char r, unsigned char g, unsigned char b)
    {
        gpuComm->setBgColor(make_float4(r/255.f,g/255.f,b/255.f,1.f));
    }
};

#endif
