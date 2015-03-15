#include "transfer_func1d.h"
#include "ui_transfer_func1d.h"


Transfer_Func1D::Transfer_Func1D(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::Transfer_Func1D)
{
	ui->setupUi(this);	

	binnumber = 256;
	ui->binsize_textEdit->setText(QString::number(binnumber));
	
	hist_exist = false;

    ui->tranfunc_widget->Init_TransFunc(); //compute initial curve

    QObject::connect(ui->tranfunc_widget, SIGNAL(trfn_changed(vector<float> &, vector<float> &)),
                     this, SLOT(on_trfn_changed(vector<float> &, vector<float> &)) );
}

Transfer_Func1D::~Transfer_Func1D()
{
    delete ui;
}

void Transfer_Func1D::on_binsize_textEdit_textChanged()
{
	QString str;
	str.append(ui->binsize_textEdit->toPlainText());
	binnumber = str.toInt();
	//binWidth = str.toFloat();
}
//test

void Transfer_Func1D::graph_show()
{
	
    ui->tranfunc_widget->updateGL();
}

void Transfer_Func1D::compute_histogram()
{	
#if 0
	glwtrf->histc = info->histogram(binWidth,g.data.minvalue,g.data.maxvalue);
	glwtrf->hist_gridsize = info->hist_gridsize;
	hist_exist = true;
#endif
}

void Transfer_Func1D::on_Binsize_pushButton_clicked()
{
	if(data_exist)
	{
		compute_histogram();
        ui->tranfunc_widget->updateGL();
	}
}

void Transfer_Func1D::on_hist_scale_verticalSlider_valueChanged(int value)
{
    ui->tranfunc_widget->hist_scale = (float)(value);
    ui->tranfunc_widget->updateGL();
}

void Transfer_Func1D::on_hist_brightness_horizontalSlider_valueChanged(int value)
{
    ui->tranfunc_widget->hist_brightness = (float)value/100.0f; //maximum = 1000;
    ui->tranfunc_widget->updateGL();
}

void Transfer_Func1D::on_Curve_radioButton_clicked(bool checked)
{
	if(checked)
	{
        if((ui->tranfunc_widget->curve_transfeFunc)==false)
		{
            ui->tranfunc_widget->curve_transfeFunc = true;
            ui->tranfunc_widget->Recal_CtrP();
		}
	}
	else
        ui->tranfunc_widget->curve_transfeFunc = false;
	
    ui->tranfunc_widget->updateGL();
}

void Transfer_Func1D::on_StraightLine_radioButton_clicked(bool checked)
{
	if(checked)
	{
        if(ui->tranfunc_widget->curve_transfeFunc)
		{
            ui->tranfunc_widget->curve_transfeFunc = false;
            ui->tranfunc_widget->Recal_CtrP();
		}
	}
	else
        ui->tranfunc_widget->curve_transfeFunc = true;

    ui->tranfunc_widget->updateGL();
}

void Transfer_Func1D::on_changed_value()
{
    //float minValue = glwtrf->glw->minValue;
    //float maxValue = glwtrf->glw->maxValue;
    //float ValueRange = glwtrf->glw->ValueRange;

    //float iso = (glwtrf->pointx)*ValueRange+minValue;

    //iso = iso>=minValue ? iso:minValue;
    //iso = iso<=maxValue ? iso:maxValue;

    //ui->show_iso_textEdit->setText(QString::number(iso));
    //ui->show_alpha_textEdit->setText(QString::number(glwtrf->alphavalue));
}

#if 0
void Transfer_Func1D::compute_skewkurtosis(int option)
{
	stsi->hist_gridsize = glwtrf->hist_gridsize;
	moment_w = 5; //must be odd
	glwtrf->moment_matrx = stsi->construct_moment_matrx(moment_w,option);
	glwtrf->moment_width = stsi->hist_gridsize;
	glwtrf->moment_height = stsi->moment_height;
}

void Transfer_Func1D::compute_moment(int degree)
{
	stsi->hist_gridsize = glwtrf->hist_gridsize;
	moment_w = 5; //must be odd
	glwtrf->moment_matrx = stsi->construct_moment_matrx(moment_w,2,degree);
	glwtrf->moment_width = stsi->hist_gridsize;
	glwtrf->moment_height = stsi->moment_height;
}
#endif

void Transfer_Func1D::on_ScalarV_comboBox_currentIndexChanged(int index)
{
}

void Transfer_Func1D::closeEvent( QCloseEvent * event )
{
	emit tr1D_window_closed();
}

void Transfer_Func1D::on_comboBox_currentIndexChanged(int index)
{
    ui->tranfunc_widget->color_type = index;
    ui->tranfunc_widget->Init_TransFunc();

//	ui->tranfunc_widget->glw->Change_Color();
    ui->tranfunc_widget->updateGL();
//	ui->tranfunc_widget->glw->updateGL();
}

void Transfer_Func1D::on_comboBox_activated(const QString &arg1)
{

}

void Transfer_Func1D::on_comboBox_currentIndexChanged(const QString &arg1)
{

}

void Transfer_Func1D::on_trfn_changed(vector<float> &color, vector<float> &alpha) {
    std::cout << ("trfn changed!!");
    emit trfn_changed(color, alpha);
}


void Transfer_Func1D:: emitTranFunc()
{
    ui->tranfunc_widget->emitTranFunc();
}
