#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

String face_cascade_name = "res/haarcascade_frontalface_alt.xml";

CascadeClassifier face_cascade;

/* Recherche de visages dans une image */
std::vector<Rect> rechercheVisages(Mat image, float scaleFactor, int minNeighbors);

int main (int argc,  char** argv) {
  // Nom de l'image à traiter
  String nomImageSource ;
  String nomImageCible ;
  String window_name = "Capture - reconnaissance de visages";
  // Image source, modifiée
  Mat imageSource;
  Mat imagePresentation;
  // Image de chaque visage
  Mat imageVisage;

  std::vector<Rect> faces;

  Scalar couleur_titre = Scalar(255,255,255);
  Scalar couleur_visage = Scalar(255,0,255);

  // Paramètres de détection des visages
  float scaleFactor=1.1;
  int minNeighbors=3;

  // Vérification des arguments de la ligne de commande
  if (argc < 3) {
    std::cout << "Merci de renseigner le nom de l'image à analyser et le nom de l'image cible dans la ligne de commande" << '\n';
    std::cout << "Syntaxe : findface <image source> <image cible> (scaleFactor minNeighbors)" << '\n';
    exit(-1);
  } else {
    if (argc > 3) {
      scaleFactor = atof(argv[3]);
      minNeighbors = atoi(argv[4]);
    }
    // Vérification de l'existance de l'image source
    nomImageSource = argv[1];
    nomImageCible = argv[2];
    std::cout << "Image à traiter : " << nomImageSource << " -> " << nomImageCible << '\n';
    printf("%fr - %d\n", scaleFactor, minNeighbors);
  }

  // Chargement des caractéristiques des visages recherchés
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

  imageSource = imread(nomImageSource, CV_LOAD_IMAGE_COLOR);
  imageSource.copyTo(imagePresentation);

  faces = rechercheVisages(imageSource, scaleFactor, minNeighbors);

  // Pour chaque visage trouvé
  for (size_t i = 0; i < faces.size(); i++) {
    // Marquage du visage
    Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    Point txt_center( faces[i].x + faces[i].width/2, faces[i].y );
    Point pt1 (faces[i].x, faces[i].y );
    Point pt2 (faces[i].x+faces[i].width, faces[i].y + faces[i].height);
    Point pt_text (faces[i].x, faces[i].y-10 );

    putText(imagePresentation, "Visage "+ std::to_string(i), pt_text, FONT_HERSHEY_SIMPLEX, 0.5, couleur_visage, 2);
    //ellipse( webcamImage, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, couleur_visage, 4, 8, 0);
    rectangle(imagePresentation, pt1, pt2, couleur_visage, 4, 8, 0);
  }
  // Affichage dans l'image du nombre de visages détectés
  putText(imagePresentation, "Nombre de visages : " + std::to_string (faces.size()), Point(10,25), FONT_HERSHEY_SIMPLEX, 1, couleur_titre, 2);

  std::cout << std::to_string (faces.size()) << " visages" << '\n';
  imwrite(nomImageCible, imagePresentation);
}

/* Recherche de visages dans une image */
std::vector<Rect> rechercheVisages(Mat image, float scaleFactor, int minNeighbors) {
  std::vector<Rect> faces;
  Mat simpleImage;

  // Simplication de l'image
  cvtColor(image, simpleImage, CV_BGR2GRAY);
  equalizeHist(simpleImage, simpleImage);

  // Recherche de visages
  face_cascade.detectMultiScale(simpleImage, faces, scaleFactor, minNeighbors, CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30), Size(200,200));

  return faces;
}
