// rvc-ui — Qt6 Widgets front-end for the RVC-GGML toolchain.
#include <QApplication>

#include "main_window.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("rvc-ui");
    QApplication::setApplicationDisplayName("RVC-GGML UI");
    QApplication::setOrganizationName("RVC-GGML");

    MainWindow window;
    window.resize(1020, 760);
    window.show();
    return app.exec();
}
