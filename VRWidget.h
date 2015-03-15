/////////////////////////////////////////////////////
/// The volume rendering widget on Qt
/// Chun-Ming Chen
/////////////////////////////////////////////////////

#ifndef VRWIDGE_HEADER_H
#define VRWIDGE_HEADER_H

#include <cpp_headers.h>
#include "CudaGLBase.h"
#include "GLVTKWidget.h"
#include "vtk_headers.h"
#include "transfer_func1d.h"

#define TrFn_WIDTH 256

using namespace std;

class VRWidget : public GLVTKWidget
{
    Q_OBJECT

public:
    bool draw_boundbox = false;

    VRWidget(QWidget *parent);
    virtual ~VRWidget();

    //virtual void paintGL();
    virtual void initializeGL();
    virtual void resizeGL(int w,int h);
    //virtual void mousePressEvent(QMouseEvent *event);
    //virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void keyPressEvent(QKeyEvent *);


    void setData(vtkSmartPointer<vtkImageData> data, vtkSmartPointer<vtkImageData> label);

    void setTrFn2D();

    void removeData() {//todo
        g_releaseTrFn();
    }

public slots:

    void on_trfn_changed(vector<float> &color, vector<float> &alpha);

protected:
    CudaGLBase *gpuComm;
    Transfer_Func1D *trfn_window;

    bool initialized = false;
    
    bool trfn_dirty = true;
    vector<float4> trfn_table;
    int trfn_table_xdim=0;

    virtual void opengl_draw();

    inline void setBgColor(unsigned char r, unsigned char g, unsigned char b)
    {
        gpuComm->setBgColor(make_float4(r/255.f,g/255.f,b/255.f,1.f));
    }

    void uploadTrFn2D();

};

#endif
