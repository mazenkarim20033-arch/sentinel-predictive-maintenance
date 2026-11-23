export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.status = status;
  }
}
