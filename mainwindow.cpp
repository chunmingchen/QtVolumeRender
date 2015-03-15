#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "vtk_headers.h"
#include "cpp_headers.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnLoad_clicked()
{
    printf("Loading file %d\n", this->file_idx);

    vtkNew<vtkXMLImageDataReader> reader1, reader2;
    vtkNew<vtkImageCast> caster1, caster2;

    stringstream ss;
    ss << "/data/flow2/soumya/raw_data/tornado_lambda2_" << file_idx << ".vti";

    /// data
    reader1->SetFileName(ss.str().c_str());
    reader1->Update();
    this->data = reader1->GetOutput();
                                                                //reader->SetFileName("/data/flow2/isabel_all/Pf05.binLE.raw_corrected_normalized.vti");
                                                                //caster1->SetInputConnection(reader1->GetOutputPort());
                                                                //caster1->SetOutputScalarTypeToFloat();
                                                                //caster1->Update();

                                                                //this->data->PrintSelf(cout, vtkIndent());

    /// label
    ss.str("");
    ss << "/data/flow2/soumya/visibility_fields/combined_final" << file_idx << ".vti";

    reader2->SetFileName(ss.str().c_str());
    reader2->Update();
    this->label = reader2->GetOutput();
                                                                //caster2->SetInputConnection(reader2->GetOutputPort());
                                                                //caster2->SetOutputScalarTypeToFloat();

    /// set data
    VRWidget *vrwidget = ui->widget;  // look into ui_mainwindow.h
    vrwidget->setData(data, label);

    vrwidget->updateGL();
}

void MainWindow::on_btnNext_clicked()
{
    if (file_idx < 10) {
        file_idx ++;
        on_btnLoad_clicked();
    }

}

void MainWindow::on_btnPrev_clicked()
{
    if (file_idx > 1) {
        file_idx --;
        on_btnLoad_clicked();
    }
}
