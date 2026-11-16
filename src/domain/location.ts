export interface Location {
  country: "CH";

  postalCode: string;
  additionalDigit: string;

  name: string;
  municipality: string;

  bfsNumber: number;
  canton: string;

  latitude: number;
  longitude: number;
}