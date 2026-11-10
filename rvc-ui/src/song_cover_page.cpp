// song_cover_page.cpp — one-click song-cover flow.
//
// Pipeline (each external step runs rvc-sep / RVC-GGML / sox as a subprocess):
//   1. rvc-sep <vocal-separator>  song.wav    -> vocals (+ instrumental)
//   2. optional: rvc-sep <backup-separator>   vocals -> main + background stems
//   3. RVC-GGML conversion of the main vocals (and optionally the background)
//   4. optional: sox/ffmpeg pitch-shift of the instrumental (same amount)
//   5. in-process mix (background + main + instrumental gains) -> output WAV

#include "song_cover_page.h"

#include "audio_utils.h"
#include "paths.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollBar>
#include <QSpinBox>
#include <QStandardPaths>
#include <QToolButton>
#include <QVBoxLayout>

namespace {

QString display_name(const QString& path) {
    return QFileInfo(path).fileName();
}

} // namespace

SongCoverPage::SongCoverPage(QWidget* parent) : QWidget(parent) {
    build_ui();
    populate_models();
}

SongCoverPage::~SongCoverPage() {
    if (process_ != nullptr) {
        process_->kill();
        process_->waitForFinished(2000);
    }
}

void SongCoverPage::build_ui() {
    auto* layout = new QVBoxLayout(this);

    // ---------------- Source / target -------------------------------------
    auto* io_group = new QGroupBox(QStringLiteral("Song and voice"), this);
    auto* io_form = new QFormLayout(io_group);

    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        song_edit_ = new QLineEdit(row);
        song_edit_->setPlaceholderText(QStringLiteral("path to a song file (WAV/FLAC/MP3/OGG)"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, &SongCoverPage::browse_song);
        h->addWidget(song_edit_, 1);
        h->addWidget(browse);
        io_form->addRow(QStringLiteral("Song"), row);
    }
    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        voice_combo_ = new QComboBox(row);
        auto* refresh = new QPushButton(QStringLiteral("Refresh"), row);
        connect(refresh, &QPushButton::clicked, this, &SongCoverPage::refresh_models);
        h->addWidget(voice_combo_, 1);
        h->addWidget(refresh);
        io_form->addRow(QStringLiteral("Voice model"), row);
    }
    {
        auto* row = new QWidget(io_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        output_edit_ = new QLineEdit(row);
        output_edit_->setPlaceholderText(QStringLiteral("output WAV file"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, &SongCoverPage::browse_output);
        h->addWidget(output_edit_, 1);
        h->addWidget(browse);
        io_form->addRow(QStringLiteral("Output"), row);
    }
    layout->addWidget(io_group);

    // ---------------- Models ----------------------------------------------
    auto* models_group = new QGroupBox(QStringLiteral("Models"), this);
    auto* models_form = new QFormLayout(models_group);

    vocal_sep_combo_ = new QComboBox(models_group);
    models_form->addRow(QStringLiteral("Vocal separator"), vocal_sep_combo_);

    {
        auto* row = new QWidget(models_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        split_backup_check_ = new QCheckBox(QStringLiteral("Separate background singers"), row);
        backup_sep_combo_ = new QComboBox(row);
        backup_sep_combo_->setEnabled(false);
        connect(split_backup_check_, &QCheckBox::toggled, backup_sep_combo_, &QComboBox::setEnabled);
        h->addWidget(split_backup_check_);
        h->addWidget(backup_sep_combo_, 1);
        models_form->addRow(QStringLiteral("Background"), row);
    }

    extractor_combo_ = new QComboBox(models_group);
    models_form->addRow(QStringLiteral("Feature extractor"), extractor_combo_);
    f0_combo_ = new QComboBox(models_group);
    models_form->addRow(QStringLiteral("F0 extractor"), f0_combo_);

    layout->addWidget(models_group);

    // ---------------- Options ----------------------------------------------
    auto* options_group = new QGroupBox(QStringLiteral("Options"), this);
    auto* options_form = new QFormLayout(options_group);

    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        octaves_spin_ = new QSpinBox(row);
        octaves_spin_->setRange(-2, 2);
        octaves_spin_->setSuffix(QStringLiteral(" octaves"));
        semitones_spin_ = new QSpinBox(row);
        semitones_spin_->setRange(-12, 12);
        semitones_spin_->setSuffix(QStringLiteral(" semitones"));
        h->addWidget(octaves_spin_);
        h->addWidget(semitones_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("Pitch shift"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        main_gain_spin_ = new QDoubleSpinBox(row);
        main_gain_spin_->setRange(-40.0, 20.0);
        main_gain_spin_->setSuffix(QStringLiteral(" dB"));
        inst_gain_spin_ = new QDoubleSpinBox(row);
        inst_gain_spin_->setRange(-40.0, 20.0);
        inst_gain_spin_->setSuffix(QStringLiteral(" dB"));
        backup_gain_spin_ = new QDoubleSpinBox(row);
        backup_gain_spin_->setRange(-40.0, 20.0);
        backup_gain_spin_->setSuffix(QStringLiteral(" dB"));
        h->addWidget(new QLabel(QStringLiteral("main"), row));
        h->addWidget(main_gain_spin_);
        h->addWidget(new QLabel(QStringLiteral("inst"), row));
        h->addWidget(inst_gain_spin_);
        h->addWidget(new QLabel(QStringLiteral("background"), row));
        h->addWidget(backup_gain_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("Mix gains"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        index_rate_spin_ = new QDoubleSpinBox(row);
        index_rate_spin_->setRange(0.0, 1.0);
        index_rate_spin_->setSingleStep(0.05);
        index_rate_spin_->setValue(0.75);
        protect_spin_ = new QDoubleSpinBox(row);
        protect_spin_->setRange(0.0, 0.5);
        protect_spin_->setSingleStep(0.01);
        protect_spin_->setValue(0.33);
        h->addWidget(new QLabel(QStringLiteral("index rate"), row));
        h->addWidget(index_rate_spin_);
        h->addWidget(new QLabel(QStringLiteral("protect"), row));
        h->addWidget(protect_spin_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("RVC"), row);
    }
    {
        auto* row = new QWidget(options_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        rms_mix_spin_ = new QDoubleSpinBox(row);
        rms_mix_spin_->setRange(0.0, 1.0);
        rms_mix_spin_->setSingleStep(0.05);
        rms_mix_spin_->setValue(1.0);
        sid_spin_ = new QSpinBox(row);
        sid_spin_->setRange(0, 500);
        device_combo_ = new QComboBox(row);
        device_combo_->addItem(QStringLiteral("GPU (Vulkan)"), 0);
        device_combo_->addItem(QStringLiteral("CPU"), 99);
        output_sr_combo_ = new QComboBox(row);
        output_sr_combo_->addItem(QStringLiteral("32000 Hz"), 32000);
        output_sr_combo_->addItem(QStringLiteral("40000 Hz"), 40000);
        output_sr_combo_->addItem(QStringLiteral("44100 Hz"), 44100);
        output_sr_combo_->addItem(QStringLiteral("48000 Hz"), 48000);
        output_sr_combo_->setCurrentIndex(2);
        h->addWidget(new QLabel(QStringLiteral("rms mix"), row));
        h->addWidget(rms_mix_spin_);
        h->addWidget(new QLabel(QStringLiteral("speaker"), row));
        h->addWidget(sid_spin_);
        h->addWidget(new QLabel(QStringLiteral("device"), row));
        h->addWidget(device_combo_);
        h->addWidget(new QLabel(QStringLiteral("output rate"), row));
        h->addWidget(output_sr_combo_);
        h->addStretch(1);
        options_form->addRow(QStringLiteral("More"), row);
    }
    layout->addWidget(options_group);

    // ---------------- Run ---------------------------------------------------
    auto* run_group = new QGroupBox(QStringLiteral("Run"), this);
    auto* run_layout = new QVBoxLayout(run_group);

    steps_list_ = new QListWidget(run_group);
    steps_list_->setMinimumHeight(120);
    run_layout->addWidget(steps_list_);

    log_view_ = new QPlainTextEdit(run_group);
    log_view_->setReadOnly(true);
    log_view_->setMinimumHeight(180);
    run_layout->addWidget(log_view_, 1);

    {
        auto* row = new QWidget(run_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        run_button_ = new QPushButton(QStringLiteral("Generate song cover"), row);
        run_button_->setDefault(true);
        cancel_button_ = new QPushButton(QStringLiteral("Cancel"), row);
        cancel_button_->setEnabled(false);
        connect(run_button_, &QPushButton::clicked, this, &SongCoverPage::start_run);
        connect(cancel_button_, &QPushButton::clicked, this, &SongCoverPage::cancel_run);
        h->addStretch(1);
        h->addWidget(run_button_);
        h->addWidget(cancel_button_);
        run_layout->addWidget(row);
    }
    layout->addWidget(run_group, 1);
}

void SongCoverPage::populate_models() {
    const QString root = paths::root();
    const QString models = paths::models_root();

    auto fill = [](QComboBox* combo, const QStringList& files, const QString& preferred) {
        combo->clear();
        for (const QString& f : files) {
            combo->addItem(display_name(f), f);
        }
        const QString pick = paths::preferred(files, preferred);
        if (!pick.isEmpty()) {
            const int idx = combo->findData(pick);
            if (idx >= 0) {
                combo->setCurrentIndex(idx);
            }
        }
        combo->setEnabled(combo->count() > 0);
    };

    fill(vocal_sep_combo_, paths::gguf_files(models + QStringLiteral("/roformers/vocal_separator")), QStringLiteral("FV4"));
    fill(backup_sep_combo_, paths::gguf_files(models + QStringLiteral("/roformers/backup_separator")), QStringLiteral("Male-Female"));
    fill(extractor_combo_, paths::gguf_files(models + QStringLiteral("/extractors")), QStringLiteral("contentvec"));
    fill(f0_combo_, paths::gguf_files(models + QStringLiteral("/f0")), QStringLiteral("rmvpe"));
    fill(voice_combo_, paths::voice_models(), QString());

    tool_sep_ = paths::tool(QStringLiteral("rvc-sep"));
    tool_rvc_ = paths::tool(QStringLiteral("RVC-GGML"));
    tool_sox_ = QStandardPaths::findExecutable(QStringLiteral("sox"));
    tool_ffmpeg_ = QStandardPaths::findExecutable(QStringLiteral("ffmpeg"));

    log_view_->clear();
    if (root.isEmpty()) {
        log(QStringLiteral("Could not locate the models/ directory. Set the RVC_ROOT environment variable to the repository root."));
    } else {
        log(QStringLiteral("Repository root: %1").arg(root));
    }
    log(QStringLiteral("Tools: rvc-sep=%1 | RVC-GGML=%2 | sox=%3")
            .arg(tool_sep_.isEmpty() ? QStringLiteral("<missing>") : tool_sep_)
            .arg(tool_rvc_.isEmpty() ? QStringLiteral("<missing>") : tool_rvc_)
            .arg(tool_sox_.isEmpty() ? QStringLiteral("<missing>") : tool_sox_));
}

void SongCoverPage::refresh_models() {
    populate_models();
}

void SongCoverPage::browse_song() {
    const QString file = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open song"), song_edit_->text(), QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (file.isEmpty()) {
        return;
    }
    song_edit_->setText(file);
    if (output_edit_->text().isEmpty()) {
        const QFileInfo fi(file);
        output_edit_->setText(fi.absolutePath() + QStringLiteral("/") + fi.completeBaseName() +
                              QStringLiteral("_cover.wav"));
    }
}

void SongCoverPage::browse_output() {
    const QString file = QFileDialog::getSaveFileName(
        this, QStringLiteral("Output file"), output_edit_->text(), QStringLiteral("Audio (*.wav *.flac *.mp3 *.ogg)"));
    if (!file.isEmpty()) {
        output_edit_->setText(file);
    }
}

void SongCoverPage::log(const QString& text) {
    QString normalized = text;
    normalized.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    const QStringList lines = normalized.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString& line : lines) {
        log_view_->appendPlainText(line);
    }
    log_view_->verticalScrollBar()->setValue(log_view_->verticalScrollBar()->maximum());
}

void SongCoverPage::set_step_state(int index, const QChar& mark) {
    if (index < 0 || index >= steps_list_->count()) {
        return;
    }
    QListWidgetItem* item = steps_list_->item(index);
    QString text = item->text();
    text.remove(0, 2);  // strip the two-character prefix
    item->setText(QStringLiteral("%1 %2").arg(mark, text));
}

int SongCoverPage::total_pitch() const {
    return octaves_spin_->value() * 12 + semitones_spin_->value();
}

QString SongCoverPage::index_path_for(const QString& voice_model) const {
    const QStringList siblings = paths::gguf_files(QFileInfo(voice_model).absolutePath());
    for (const QString& s : siblings) {
        if (s.endsWith(QStringLiteral(".index.gguf")) &&
            QFileInfo(s).completeBaseName() == QFileInfo(voice_model).completeBaseName() + QStringLiteral(".index")) {
            return s;
        }
    }
    return QString();
}

QStringList SongCoverPage::rvc_args(const QString& in, const QString& out) const {
    QStringList a;
    a << QStringLiteral("--input") << in
      << QStringLiteral("--output") << out
      << QStringLiteral("--model") << voice_combo_->currentData().toString();
    const QString idx = index_path_for(voice_combo_->currentData().toString());
    if (!idx.isEmpty()) {
        a << QStringLiteral("--index") << idx
          << QStringLiteral("--index-rate") << QString::number(index_rate_spin_->value());
    }
    a << QStringLiteral("--pitch") << QString::number(total_pitch())
      << QStringLiteral("--sid") << QString::number(sid_spin_->value())
      << QStringLiteral("--protect") << QString::number(protect_spin_->value())
      << QStringLiteral("--rms-mix-rate") << QString::number(rms_mix_spin_->value())
      << QStringLiteral("--hubert") << extractor_combo_->currentData().toString()
      << QStringLiteral("--rmvpe") << f0_combo_->currentData().toString()
      << QStringLiteral("--device") << device_combo_->currentData().toString();
    return a;
}

void SongCoverPage::start_run() {
    if (running_) {
        return;
    }
    const QString song = song_edit_->text();
    const QString output = output_edit_->text();
    if (song.isEmpty() || !QFile::exists(song)) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Pick an existing song file first."));
        return;
    }
    if (output.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Pick an output file first."));
        return;
    }
    if (tool_sep_.isEmpty() || tool_rvc_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("Song cover"),
            QStringLiteral("rvc-sep / RVC-GGML were not found next to rvc-ui.\n"
                           "Set the RVC_TOOLS_DIR environment variable to the directory that contains them."));
        return;
    }
    if (!QDir().mkpath(QFileInfo(output).absolutePath())) {
        QMessageBox::warning(this, QStringLiteral("Song cover"), QStringLiteral("Cannot create the output directory."));
        return;
    }

    work_dir_ = QFileInfo(output).absolutePath() + QStringLiteral("/") +
                QFileInfo(song).completeBaseName() + QStringLiteral("_work");
    QDir().mkpath(work_dir_);

    vocals_wav_ = work_dir_ + QStringLiteral("/vocals.wav");
    instrumental_wav_ = work_dir_ + QStringLiteral("/instrumental.wav");
    conv_main_wav_ = work_dir_ + QStringLiteral("/vocals_converted.wav");
    conv_backup_wav_ = work_dir_ + QStringLiteral("/background_converted.wav");
    shifted_instrumental_wav_ = work_dir_ + QStringLiteral("/instrumental_shifted.wav");

    plan_.clear();
    steps_list_->clear();
    plan_ << StepKind::Separate;
    steps_list_->addItem(QStringLiteral("\u00b7 Separate vocals and instrumental"));
    if (split_backup_check_->isChecked()) {
        plan_ << StepKind::SplitBackup;
        steps_list_->addItem(QStringLiteral("\u00b7 Split background singers"));
    }
    plan_ << StepKind::ConvertMain;
    steps_list_->addItem(QStringLiteral("\u00b7 Convert vocals (RVC)"));
    if (split_backup_check_->isChecked()) {
        plan_ << StepKind::ConvertBackup;
        steps_list_->addItem(QStringLiteral("\u00b7 Convert background vocals (RVC)"));
    }
    if (total_pitch() != 0) {
        plan_ << StepKind::ShiftInstrumental;
        steps_list_->addItem(QStringLiteral("\u00b7 Pitch-shift instrumental (%1 semitones)").arg(total_pitch()));
    }
    plan_ << StepKind::Mix;
    steps_list_->addItem(QStringLiteral("\u00b7 Mix and write output"));

    running_ = true;
    cancel_requested_ = false;
    step_index_ = -1;
    run_timer_.start();
    run_button_->setEnabled(false);
    cancel_button_->setEnabled(true);

    log(QStringLiteral("=== Song cover: %1 ===").arg(QFileInfo(song).fileName()));
    if (split_backup_check_->isChecked()) {
        log(QStringLiteral("Background split is enabled: main vocals = stem 0, background = stem 1."));
    }
    run_step(0);
}

void SongCoverPage::run_step(int index) {
    step_index_ = index;
    const StepKind kind = plan_.at(index);
    set_step_state(index, QChar(0x25b8));  // running marker

    switch (kind) {
        case StepKind::Separate:
            start_process(tool_sep_, {vocal_sep_combo_->currentData().toString(),
                                      song_edit_->text(), vocals_wav_});
            break;
        case StepKind::SplitBackup:
            start_process(tool_sep_, {backup_sep_combo_->currentData().toString(),
                                      vocals_wav_, work_dir_ + QStringLiteral("/background.wav")});
            break;
        case StepKind::ConvertMain:
            start_process(tool_rvc_, rvc_args(vocals_wav_, conv_main_wav_));
            break;
        case StepKind::ConvertBackup:
            start_process(tool_rvc_, rvc_args(work_dir_ + QStringLiteral("/background_stem_1.wav"),
                                              conv_backup_wav_));
            break;
        case StepKind::ShiftInstrumental: {
            const int cents = total_pitch() * 100;
            if (!tool_sox_.isEmpty()) {
                start_process(tool_sox_, {instrumental_wav_, shifted_instrumental_wav_,
                                          QStringLiteral("pitch"), QString::number(cents)});
            } else if (!tool_ffmpeg_.isEmpty()) {
                const double ratio = std::pow(2.0, total_pitch() / 12.0);
                const QString filter = QStringLiteral("asetrate=%1*%2,aresample=%1,atempo=%3")
                                           .arg(44100)
                                           .arg(ratio)
                                           .arg(1.0 / ratio);
                start_process(tool_ffmpeg_, {QStringLiteral("-y"), QStringLiteral("-i"),
                                             instrumental_wav_, QStringLiteral("-af"), filter,
                                             shifted_instrumental_wav_});
            } else {
                fail(QStringLiteral("Pitch shift is requested but neither sox nor ffmpeg was found; "
                                    "set the pitch to 0 or install sox."));
            }
            break;
        }
        case StepKind::Mix:
            run_mix();
            break;
    }
}

void SongCoverPage::start_process(const QString& program, const QStringList& args) {
    process_ = new QProcess(this);
    process_->setProgram(program);
    process_->setArguments(args);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
        log(QString::fromLocal8Bit(process_->readAllStandardOutput()));
    });
    connect(process_, &QProcess::finished, this, &SongCoverPage::on_process_finished);
    log(QStringLiteral(">> %1 %2").arg(program, args.join(QLatin1Char(' '))));
    process_->start();
    if (!process_->waitForStarted(5000)) {
        log(QStringLiteral("Failed to start: %1").arg(program));
        fail(QStringLiteral("Could not start %1").arg(QFileInfo(program).fileName()));
    }
}

void SongCoverPage::on_process_finished(int exit_code, int exit_status) {
    QProcess* proc = process_;
    process_ = nullptr;
    if (proc != nullptr) {
        proc->deleteLater();
    }
    if (cancel_requested_) {
        set_step_state(step_index_, QChar(0x2717));
        finish_run(false);
        log(QStringLiteral("Cancelled."));
        return;
    }
    if (exit_status != QProcess::NormalExit || exit_code != 0) {
        set_step_state(step_index_, QChar(0x2717));
        fail(QStringLiteral("Step failed with exit code %1. See the log above.").arg(exit_code));
        return;
    }

    const StepKind kind = plan_.at(step_index_);
    set_step_state(step_index_, QChar(0x2713));

    if (kind == StepKind::Separate && !prepare_instrumental()) {
        return;
    }

    if (step_index_ + 1 >= plan_.size()) {
        finish_run(true);
        return;
    }
    run_step(step_index_ + 1);
}

bool SongCoverPage::prepare_instrumental() {
    const QString stem0 = work_dir_ + QStringLiteral("/vocals_stem_0.wav");
    const QString stem1 = work_dir_ + QStringLiteral("/vocals_stem_1.wav");
    const QString plain = vocals_wav_;

    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);
        if (QFile::exists(stem0) && QFile::exists(stem1)) {
            // Two-stem model (e.g. InstVoc): stem 0 = vocals, stem 1 = instrumental.
            vocals_wav_ = stem0;
            instrumental_wav_ = stem1;
            log(QStringLiteral("Separator returned two stems: using the model's instrumental."));
        } else if (QFile::exists(plain)) {
            // Vocals-only model (e.g. FV4): instrumental = song - vocals.
            log(QStringLiteral("Computing instrumental (song - vocals)..."));
            audio::Buffer song = audio::resample(audio::to_stereo(audio::load_audio(song_edit_->text().toStdString())), 44100);
            audio::Buffer voc = audio::resample(audio::to_stereo(audio::load_audio(plain.toStdString())), 44100);
            audio::Buffer inst = audio::subtract(song, voc);
            audio::save_wav(instrumental_wav_.toStdString(), inst);
        } else {
            QApplication::restoreOverrideCursor();
            fail(QStringLiteral("Separation produced no output file."));
            return false;
        }
        QApplication::restoreOverrideCursor();
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        fail(QString::fromStdString(e.what()));
        return false;
    }
    return true;
}

void SongCoverPage::run_mix() {
    try {
        QApplication::setOverrideCursor(Qt::WaitCursor);

        const bool shifted_this_run =
            plan_.contains(StepKind::ShiftInstrumental) && QFile::exists(shifted_instrumental_wav_);
        const QString instrumental_file = shifted_this_run
            ? shifted_instrumental_wav_ : instrumental_wav_;
        audio::Buffer instrumental = audio::resample(audio::to_stereo(audio::load_audio(instrumental_file.toStdString())), 44100);
        audio::Buffer main = audio::resample(audio::to_stereo(audio::load_audio(conv_main_wav_.toStdString())), 44100);

        std::vector<std::pair<audio::Buffer, float>> tracks;
        tracks.emplace_back(std::move(instrumental), audio::db_to_linear(static_cast<float>(inst_gain_spin_->value())));
        tracks.emplace_back(std::move(main), audio::db_to_linear(static_cast<float>(main_gain_spin_->value())));
        if (split_backup_check_->isChecked() && QFile::exists(conv_backup_wav_)) {
            audio::Buffer backup = audio::resample(audio::to_stereo(audio::load_audio(conv_backup_wav_.toStdString())), 44100);
            tracks.emplace_back(std::move(backup), audio::db_to_linear(static_cast<float>(backup_gain_spin_->value())));
        }

        audio::Buffer mixed = audio::mix(tracks);
        const float norm_db = audio::apply_peak_normalization(mixed);
        if (norm_db != 0.0f) {
            log(QStringLiteral("Peak normalization: %1%2 dB (to -1 dBFS)")
                    .arg(norm_db > 0.0f ? QStringLiteral("+") : QString())
                    .arg(static_cast<double>(norm_db), 0, 'f', 1));
        }
        const int out_sr = output_sr_combo_->currentData().toInt();
        if (out_sr != 44100) {
            mixed = audio::resample(mixed, out_sr);
        }

        const QString target = output_edit_->text();
        if (target.endsWith(QStringLiteral(".wav"), Qt::CaseInsensitive)) {
            audio::save_wav(target.toStdString(), mixed);
        } else {
            const QString wav_path = work_dir_ + QStringLiteral("/mix.wav");
            audio::save_wav(wav_path.toStdString(), mixed);
            QString err;
            if (!encode_to(wav_path, target, &err)) {
                QApplication::restoreOverrideCursor();
                set_step_state(step_index_, QChar(0x2717));
                fail(err);
                return;
            }
        }

        QApplication::restoreOverrideCursor();
        set_step_state(step_index_, QChar(0x2713));
        log(QStringLiteral("Wrote %1 (%2 s @ %3 Hz)")
                .arg(target)
                .arg(static_cast<double>(mixed.frames()) / mixed.sample_rate, 0, 'f', 2)
                .arg(mixed.sample_rate));
        finish_run(true);
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        set_step_state(step_index_, QChar(0x2717));
        fail(QString::fromStdString(e.what()));
    }
}

bool SongCoverPage::encode_to(const QString& wav_path, const QString& target_path, QString* err) {
    if (tool_ffmpeg_.isEmpty()) {
        *err = QStringLiteral("ffmpeg was not found; it is required for FLAC/MP3/OGG output");
        return false;
    }
    QStringList args;
    args << QStringLiteral("-y") << QStringLiteral("-i") << wav_path;
    const QString lower = target_path.toLower();
    if (lower.endsWith(QStringLiteral(".flac"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("flac");
    } else if (lower.endsWith(QStringLiteral(".mp3"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("libmp3lame")
             << QStringLiteral("-q:a") << QStringLiteral("2");
    } else if (lower.endsWith(QStringLiteral(".ogg"))) {
        args << QStringLiteral("-c:a") << QStringLiteral("libvorbis")
             << QStringLiteral("-q:a") << QStringLiteral("5");
    } else {
        *err = QStringLiteral("Unsupported output extension (use .wav, .flac, .mp3 or .ogg)");
        return false;
    }
    args << target_path;
    log(QStringLiteral(">> %1 %2").arg(tool_ffmpeg_, args.join(QLatin1Char(' '))));

    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.start(tool_ffmpeg_, args);
    if (!proc.waitForFinished(600000)) {
        *err = QStringLiteral("ffmpeg did not finish in time");
        return false;
    }
    const QByteArray out = proc.readAllStandardOutput();
    if (!out.isEmpty()) {
        log(QString::fromLocal8Bit(out));
    }
    if (proc.exitStatus() != QProcess::NormalExit || proc.exitCode() != 0) {
        *err = QStringLiteral("ffmpeg failed (exit code %1)").arg(proc.exitCode());
        return false;
    }
    return true;
}

void SongCoverPage::fail(const QString& message) {
    log(QStringLiteral("[error] %1").arg(message));
    finish_run(false);
}

void SongCoverPage::finish_run(bool success) {
    running_ = false;
    cancel_requested_ = false;
    run_button_->setEnabled(true);
    cancel_button_->setEnabled(false);
    if (success) {
        log(QStringLiteral("Song cover generated successfully."));
        log(QStringLiteral("Total time: %1 s (start -> file written)")
                .arg(run_timer_.elapsed() / 1000.0, 0, 'f', 1));
    }
}

void SongCoverPage::cancel_run() {
    if (!running_) {
        return;
    }
    cancel_requested_ = true;
    if (process_ != nullptr) {
        process_->kill();
    }
}
